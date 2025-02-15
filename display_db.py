
from replit import db

def display_db_contents():
    print("Replit DB Contents:")
    print("-" * 50)
    
    if not db.keys():
        print("Database is empty")
        return
        
    for key in db.keys():
        print(f"\nKey: {key}")
        print("Value:", db[key])
        print("-" * 50)

if __name__ == "__main__":
    display_db_contents()
