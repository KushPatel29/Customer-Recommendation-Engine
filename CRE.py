import os
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

###########################
# OPTIONAL: NLTK FOR BOW
###########################
import nltk
from nltk.corpus import stopwords
# Make sure you have installed NLTK and run:
# nltk.download("stopwords")

from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.metrics.pairwise import cosine_similarity

###########################
# 1) WEB SCRAPING
###########################

def scrape_customer_details(customer_name):
    """
    Scrape Bing for basic business info about 'customer_name'.
    Returns a dict with "titles" and "snippets" (top 5 each).
    """
    try:
        query = f"{customer_name.replace(' ', '+')}+business"
        url = f"https://www.bing.com/search?q={query}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"[ERROR] {customer_name}: Bing status {resp.status_code}")
            return {"titles": [], "snippets": []}

        soup = BeautifulSoup(resp.text, "html.parser")

        titles = [t.get_text(strip=True) for t in soup.find_all('h2')]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

        return {
            "titles": titles[:5],
            "snippets": paragraphs[:5]
        }
    except Exception as e:
        print(f"[ERROR] scraping '{customer_name}': {e}")
        return {"titles": [], "snippets": []}


def scrape_similar_businesses_to_customer(customer_name, region, limit=5):
    """
    Scrape Bing by searching "similar businesses to <customer_name> in <region>".
    Returns a list of dicts: [{ "title": "...", "snippet": "..." }, ...]
    """
    try:
        query = f"similar businesses to {customer_name.replace(' ', '+')} in {region.replace(' ', '+')}"
        url = f"https://www.bing.com/search?q={query}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"[ERROR] Failed to fetch leads for '{customer_name}' in {region}: status {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        titles = [t.get_text(strip=True) for t in soup.find_all('h2')]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

        count = min(limit, len(titles), len(paragraphs))
        leads = []
        for i in range(count):
            leads.append({
                "title": titles[i],
                "snippet": paragraphs[i]
            })
        return leads

    except Exception as e:
        print(f"[ERROR] scraping similar businesses for '{customer_name}' in '{region}': {e}")
        return []


###########################
# 2) DATA ANALYSIS
###########################

def get_top_n_customers_all_regions(data, n=50):
    """
    Group by (RegionName, CustomerName) -> sum(Rev).
    For each region, pick top N customers, then unify
    them into one DataFrame.
    """
    req_cols = {"RegionName", "CustomerName", "Rev"}
    if not req_cols.issubset(data.columns):
        raise ValueError(f"Data must have at least {req_cols}.")

    grouped = (
        data.groupby(["RegionName", "CustomerName"])["Rev"]
        .sum()
        .reset_index()
        .rename(columns={"Rev": "TotalRevenue"})
        .sort_values(by=["RegionName","TotalRevenue"], ascending=[True,False])
    )

    # For each region, grab top N
    region_frames = []
    for region, gdf in grouped.groupby("RegionName"):
        top_slice = gdf.head(n)
        region_frames.append(top_slice)

    top_n_df = pd.concat(region_frames, ignore_index=True)
    return top_n_df


def recommend_new_customers(
    data: pd.DataFrame,
    top_n_df: pd.DataFrame,
    min_lineitem_revenue=500
) -> pd.DataFrame:
    """
    For each region in top_n_df, find new customers not in top 50,
    but having line-item Rev > min_lineitem_revenue.
    Return one combined DataFrame across all regions with:
       RegionName, CustomerName, TotalRevenue, OrderFrequency, TotalQuantity
    """
    needed = {"RegionName", "CustomerName", "Rev", "OrderId", "QuantityOrdered"}
    if not needed.issubset(data.columns):
        raise ValueError(f"data must have columns {needed}")

    recommended_frames = []

    # Group top_n by region so we can exclude them region by region
    for region, subtop in top_n_df.groupby("RegionName"):
        region_data = data[data["RegionName"] == region].copy()
        if region_data.empty:
            continue

        # Exclude top customers in that region
        exclude_names = subtop["CustomerName"].unique()
        region_data = region_data[~region_data["CustomerName"].isin(exclude_names)]
        if region_data.empty:
            continue

        # Filter line-item Rev
        region_data = region_data[region_data["Rev"] > min_lineitem_revenue]
        if region_data.empty:
            continue

        # Aggregate
        recs = (
            region_data.groupby(["RegionName","CustomerName"])
            .agg({
                "Rev":"sum",
                "OrderId":"nunique",
                "QuantityOrdered":"sum"
            })
            .rename(columns={
                "Rev":"TotalRevenue",
                "OrderId":"OrderFrequency",
                "QuantityOrdered":"TotalQuantity"
            })
            .reset_index()
            .sort_values(by=["TotalRevenue","OrderFrequency"], ascending=[False,False])
        )

        recommended_frames.append(recs)

    if recommended_frames:
        recommended_df = pd.concat(recommended_frames, ignore_index=True)
    else:
        recommended_df = pd.DataFrame(columns=["RegionName","CustomerName","TotalRevenue","OrderFrequency","TotalQuantity"])
    return recommended_df


###########################
# 3) TEXT-BASED SIMILARITY
###########################

def preprocess_text(text_list):
    """
    Clean / remove stopwords from a list of snippet strings.
    """
    stop_words = set(stopwords.words("english"))
    cleaned_list = []
    for raw in text_list:
        # Remove punctuation
        txt = re.sub(r'[^\w\s]', '', raw.lower())
        # Remove stopwords
        tokens = [w for w in txt.split() if w not in stop_words]
        cleaned_list.append(" ".join(tokens))
    return cleaned_list


def build_bow_similarity(top_customers_scraped):
    """
    top_customers_scraped: list of dicts like:
      [{ 
        "RegionName":..., 
        "CustomerName":..., 
        "ScrapedText": ...
      }, ...]
    Builds a bag-of-words matrix -> NxN cosine similarity among top customers.
    Returns a Pandas DataFrame (similarity matrix).
    """
    if not top_customers_scraped:
        return pd.DataFrame()

    raw_texts = [x["ScrapedText"] for x in top_customers_scraped]
    cleaned = preprocess_text(raw_texts)

    # Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleaned)
    rep = tokenizer.texts_to_matrix(cleaned, mode="count")

    # Cosine similarity
    sim_matrix = cosine_similarity(rep, rep)

    # Build the DataFrame
    row_names = [f"{x['RegionName']} - {x['CustomerName']}" for x in top_customers_scraped]
    sim_df = pd.DataFrame(sim_matrix, index=row_names, columns=row_names)
    return sim_df


###########################
# 4) MAIN PIPELINE
###########################

def run_final_pipeline(
    data: pd.DataFrame,
    top_n=50,
    min_lineitem_revenue=500,
    output_file="final_output.csv"
):
    """
    1) Find top N customers per region, combined (one table).
    2) Scrape each top customer's snippet data.
    3) Build optional text-based BOW similarity among top customers.
    4) Recommend new customers (excluded from top N, with line-item Rev > threshold).
    5) Scrape "similar businesses" for each top customer (new leads).
    6) Produce one final output (Excel with 3 sheets: Top50, Recommended, NewLeads),
       plus a single CSV that merges them all with a "Type" column.
    """

    # --- A) Top 50
    print("=== STEP A: Find Top 50 by Region ===")
    top_n_df = get_top_n_customers_all_regions(data, n=top_n)
    if top_n_df.empty:
        print("[WARNING] No top customers found. Check data!")
        return

    # --- B) Scrape each top customer
    print("\n=== STEP B: Scrape Each Top Customer ===")
    top_scrape_info = []
    leads_list = []  # for "similar businesses" leads
    for idx, row in top_n_df.iterrows():
        region = row["RegionName"]
        cname  = row["CustomerName"]
        # i) Basic info
        detail = scrape_customer_details(cname)
        combined_text = " ".join(detail["titles"] + detail["snippets"])
        top_scrape_info.append({
            "RegionName": region,
            "CustomerName": cname,
            "ScrapedText": combined_text
        })

        # ii) "Similar businesses" leads
        new_leads = scrape_similar_businesses_to_customer(cname, region, limit=5)
        for lead in new_leads:
            leads_list.append({
                "RegionName": region,
                "TopCustomerName": cname,
                "LeadTitle": lead["title"],
                "LeadSnippet": lead["snippet"]
            })

    # Deduplicate leads if needed
    leads_df = pd.DataFrame(leads_list).drop_duplicates(
        subset=["RegionName","TopCustomerName","LeadTitle","LeadSnippet"]
    )

    # --- C) Build BOW similarity for top customers (OPTIONAL)
    print("\n=== STEP C: Build BOW Similarity (Optional) ===")
    sim_df = build_bow_similarity(top_scrape_info)

    # --- D) Recommend new customers
    print("\n=== STEP D: Recommend New Customers ===")
    recommended_df = recommend_new_customers(data, top_n_df, min_lineitem_revenue=min_lineitem_revenue)

    # Merge the snippet text into top_n_df for final output
    top_scraped_df = pd.DataFrame(top_scrape_info)  # [RegionName, CustomerName, ScrapedText]
    # We'll join on (RegionName, CustomerName)
    top_n_merged = pd.merge(
        top_n_df,
        top_scraped_df,
        how="left",
        on=["RegionName","CustomerName"]
    )

    # Prepare final data for single CSV with a "Type" column
    # 1) top customers
    top_n_merged["Type"] = "Top50"
    top_n_merged.rename(columns={"TotalRevenue":"TotalRevenueTop"}, inplace=True)

    # 2) recommended
    recommended_df["Type"] = "Recommended"

    # 3) new leads
    # For new leads, let's unify them into a single table with region, lead info
    # We'll store RegionName, TopCustomerName, LeadTitle, LeadSnippet
    # plus a "Type"= "NewLead"
    if not leads_df.empty:
        leads_df["Type"] = "NewLead"

    # Single CSV approach:
    # We'll unify top customers, recommended, and leads each having some different columns.
    # We'll define a minimal set of columns for each, adding placeholders so we can unify them.

    # A) Top 50 columns
    top_n_cols = ["RegionName","CustomerName","Type","ScrapedText","TotalRevenueTop"]
    top_n_filled = top_n_merged[top_n_cols].copy()
    top_n_filled["TopCustomerName"] = None
    top_n_filled["LeadTitle"] = None
    top_n_filled["LeadSnippet"] = None
    top_n_filled["TotalRevenue"] = None
    top_n_filled["OrderFrequency"] = None
    top_n_filled["TotalQuantity"] = None

    # B) Recommended columns
    rec_cols = ["RegionName","CustomerName","Type","TotalRevenue","OrderFrequency","TotalQuantity"]
    rec_filled = recommended_df[rec_cols].copy()
    rec_filled["ScrapedText"] = None
    rec_filled["TopCustomerName"] = None
    rec_filled["LeadTitle"] = None
    rec_filled["LeadSnippet"] = None
    rec_filled["TotalRevenueTop"] = None

    # C) New Leads columns
    leads_out = ["RegionName","TopCustomerName","LeadTitle","LeadSnippet","Type"]
    # We also want "CustomerName" but that's not relevant for leads
    # We'll fill placeholders so we can unify them in a single DF
    if leads_df.empty:
        leads_filled = pd.DataFrame(columns=[
            "RegionName","CustomerName","Type","ScrapedText","TopCustomerName","LeadTitle","LeadSnippet",
            "TotalRevenueTop","TotalRevenue","OrderFrequency","TotalQuantity"
        ])
    else:
        leads_filled = leads_df[leads_out].copy()
        leads_filled["CustomerName"] = None
        leads_filled["ScrapedText"] = None
        leads_filled["TotalRevenueTop"] = None
        leads_filled["TotalRevenue"] = None
        leads_filled["OrderFrequency"] = None
        leads_filled["TotalQuantity"] = None

    # Now unify
    final_combined = pd.concat([top_n_filled, rec_filled, leads_filled], ignore_index=True)

    # We produce one Excel with 3 sheets: "Top50", "Recommended", "NewLeads"
    excel_out = output_file.replace(".csv", ".xlsx")
    with pd.ExcelWriter(excel_out) as writer:
        top_n_merged.to_excel(writer, sheet_name="Top50", index=False)
        recommended_df.to_excel(writer, sheet_name="Recommended", index=False)
        leads_df.to_excel(writer, sheet_name="NewLeads", index=False)
        # If you want the optional similarity matrix as well:
        if not sim_df.empty:
            sim_df.to_excel(writer, sheet_name="Similarity")

    print(f"\n[INFO] Wrote multi-sheet Excel -> {excel_out}")

    # Then produce one final CSV
    final_combined.to_csv(output_file, index=False)
    print(f"[INFO] Wrote single CSV with Type='Top50','Recommended','NewLead' -> {output_file}")
    print("\n=== PIPELINE COMPLETE. ===")


##################################
# 5) USAGE EXAMPLE (MAIN)
##################################
if _name_ == "_main_":
    # 1) Load your Excel file
    excel_path = r"C:\Users\Kush\Desktop\Sales\Sales_Data.xlsx"
    sales_data = pd.read_excel(excel_path, sheet_name=0)

    # 2) Run the pipeline
    #    - top_n=50  -> top 50 per region
    #    - min_lineitem_revenue=500 -> recommended if line item > 500
    #    - output_file="final_output.csv" -> single CSV + multi-sheet Excel
    run_final_pipeline(
        data=sales_data,
        top_n=50,
        min_lineitem_revenue=500,
        output_file="final_output.csv"
    )
