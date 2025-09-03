import sqlite3

conn = sqlite3.connect("file_uploads.db")
cur = conn.cursor()

# Helper to check if column exists
def column_exists(table, column):
    cur.execute(f"PRAGMA table_info({table})")
    columns = [info[1] for info in cur.fetchall()]
    return column in columns

# --- Fix uploads table ---
try:
    # Add filedata column if not exists
    if not column_exists("uploads", "filedata"):
        cur.execute("ALTER TABLE uploads ADD COLUMN filedata BLOB;")
        print("✅ 'filedata' column added successfully.")
    else:
        print("ℹ️ Column 'filedata' already exists, skipping.")

    # Add created_at column if not exists
    if not column_exists("uploads", "created_at"):
        cur.execute("ALTER TABLE uploads ADD COLUMN created_at TIMESTAMP;")
        print("✅ 'created_at' column added successfully.")
    else:
        print("ℹ️ Column 'created_at' already exists, skipping.")

    # Update rows with missing created_at
    cur.execute("UPDATE uploads SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL;")

except sqlite3.OperationalError as e:
    print(f"⚠️ Migration error: {e}")

conn.commit()
conn.close()
print("✅ Migration completed successfully.")
