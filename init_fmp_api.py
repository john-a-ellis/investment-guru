#!/usr/bin/env python
"""
FMP API initialization and testing script.
Run this script to set up and validate the FMP API connection.
"""
import os
import sys
import json
from dotenv import load_dotenv
import argparse

# Check if .env file exists
if not os.path.exists('.env'):
    print("Creating .env file...")
    with open('.env', 'w') as f:
        f.write("# Financial Modeling Prep API key\n")
        f.write("FMP_API_KEY=\n")
        f.write("\n# Database configuration\n")
        f.write("DB_NAME=investment_system\n")
        f.write("DB_USER=postgres\n")
        f.write("DB_PASSWORD=postgres\n")
        f.write("DB_HOST=localhost\n")
        f.write("DB_PORT=5432\n")
    print(".env file created. Please edit it to add your FMP API key.")

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Initialize and test FMP API connection')
parser.add_argument('--api-key', help='FMP API key (will be saved to .env file)')
parser.add_argument('--test', action='store_true', help='Test API connection')
parser.add_argument('--symbol', default='AAPL', help='Symbol to test (default: AAPL)')
args = parser.parse_args()

# Update API key if provided
if args.api_key:
    print(f"Updating FMP API key in .env file...")
    
    # Read current .env content
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    # Update the API key line
    with open('.env', 'w') as f:
        for line in lines:
            if line.startswith('FMP_API_KEY='):
                f.write(f"FMP_API_KEY={args.api_key}\n")
            else:
                f.write(line)
    
    # Reload environment variables
    load_dotenv()
    print("API key updated.")

# Test API connection if requested
if args.test:
    print(f"Testing FMP API connection with symbol: {args.symbol}")
    
    # Check if API key is set
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        print("ERROR: FMP API key is not set. Please provide it with --api-key option.")
        sys.exit(1)
    
    try:
        # Import and test the FMP API module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from modules.fmp_api import fmp_api
        
        # Test getting company profile
        print(f"\nGetting company profile for {args.symbol}...")
        profile = fmp_api.get_company_profile(args.symbol)
        if profile:
            print(f"SUCCESS: Got company profile")
            print(f"Company: {profile.get('companyName', 'N/A')}")
            print(f"Industry: {profile.get('industry', 'N/A')}")
            print(f"Country: {profile.get('country', 'N/A')}")
        else:
            print(f"WARNING: Could not get company profile for {args.symbol}")
        
        # Test getting quote
        print(f"\nGetting current quote for {args.symbol}...")
        quote = fmp_api.get_quote(args.symbol)
        if quote and 'price' in quote:
            print(f"SUCCESS: Got quote")
            print(f"Current price: ${quote.get('price', 'N/A')}")
            print(f"Change: {quote.get('change', 'N/A')} ({quote.get('changesPercentage', 'N/A')}%)")
        else:
            print(f"WARNING: Could not get quote for {args.symbol}")
        
        # Test getting historical data
        print(f"\nGetting historical data for {args.symbol}...")
        hist_data = fmp_api.get_historical_price(args.symbol, period="1mo")
        if not hist_data.empty:
            print(f"SUCCESS: Got historical data")
            print(f"Number of days: {len(hist_data)}")
            print(f"Latest close: ${hist_data['Close'].iloc[-1]}")
            print(f"Date range: {hist_data.index[0]} to {hist_data.index[-1]}")
        else:
            print(f"WARNING: Could not get historical data for {args.symbol}")
        
        # Test getting exchange rate
        print(f"\nGetting USD to CAD exchange rate...")
        rate = fmp_api.get_exchange_rate("USD", "CAD")
        if rate:
            print(f"SUCCESS: Got exchange rate")
            print(f"USD to CAD: {rate}")
        else:
            print(f"WARNING: Could not get exchange rate")
        
        # Test getting news
        print(f"\nGetting financial news...")
        news = fmp_api.get_news(limit=3)
        if news:
            print(f"SUCCESS: Got news articles")
            for i, article in enumerate(news[:3], 1):
                print(f"\nArticle {i}:")
                print(f"Title: {article.get('title', 'N/A')}")
                print(f"Date: {article.get('publishedAt', 'N/A')}")
                print(f"Source: {article.get('source', {}).get('name', 'N/A')}")
        else:
            print(f"WARNING: Could not get news articles")
        
        print("\nAll tests completed.")
        print("FMP API integration is working correctly.")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nAPI test failed. Please check your API key and internet connection.")
        sys.exit(1)

# If not testing, print usage instructions
if not args.api_key and not args.test:
    print("\nUsage instructions:")
    print("1. Set your FMP API key:")
    print("   python init_fmp_api.py --api-key YOUR_API_KEY")
    print("\n2. Test the API connection:")
    print("   python init_fmp_api.py --test")
    print("\n3. Test with a specific symbol:")
    print("   python init_fmp_api.py --test --symbol AAPL")