from data_loader import generate_sample_data
df = generate_sample_data(200)
df.to_csv('sample_logs.csv', index=False)
print("sample_logs.csv generated successfully.")
