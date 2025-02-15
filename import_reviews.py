from utils.db import MotherDuckStore
import os

def main():
    try:
        # Initialize MotherDuck store
        db = MotherDuckStore()
        
        # Import reviews from CSV
        csv_path = "attached_assets/Jewelry Store Google Map Reviews.csv"
        result = db.import_reviews_from_csv(csv_path)
        print(result)
        
    except Exception as e:
        print(f"Error importing reviews: {str(e)}")

if __name__ == "__main__":
    main()
