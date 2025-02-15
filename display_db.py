from replit import db


def delete_all_keys():
    counter = 0
    print("Deleting all keys from Replit DB...")
    print("-" * 50)

    if not db.keys():
        print("Database is already empty")
        return
    for key in db.keys():
        del db[key]
        print(f"Deleted key: {key}")
        counter += 1

    print("-" * 50)
    print("all keys have been deleted")


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
    delete_all_keys()
