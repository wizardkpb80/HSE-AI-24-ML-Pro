import requests

headers = {"Content-Type": "application/json"}

data = {
    "name": "Mahindra Xylo E4 BS IV",
    "year": 2010,
    "selling_price": 229999,
    "km_driven": 168000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "14 kmpl",
    "engine": "2498",
    "max_power": "112 bhp",
    "torque": "260 Nm at 1800-2200 rpm",
    "seats": 7.0
}

data2 = {
    "objects": [
        {
            "name": "Mahindra Xylo E4 BS IV",
            "year": 2010,
            "selling_price": 229999,
            "km_driven": 168000,
            "fuel": "Diesel",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
            "mileage": "14 kmpl",
            "engine": "2498",
            "max_power": "112 bhp",
            "torque": "260 Nm at 1800-2200 rpm",
            "seats": 7.0
        },
        {
            "name": "Mahindra Xylo E4 BS IV",
            "year": 2015,
            "selling_price": 229999,
            "km_driven": 68000,
            "fuel": "Diesel",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
            "mileage": "14 kmpl",
            "engine": "2498",
            "max_power": "500 bhp",
            "torque": "160 Nm at 1800-2200 rpm",
            "seats": 7.0
        }
    ]
}
url = "http://localhost:8000/test"
response = requests.post(url, headers=headers, json=data)
print(f"{url} API Response:{response.status_code} JSON Response {response.json()}")

url = "http://localhost:8000/predict_item"
response = requests.post(url, headers=headers, json=data)
print(f"{url} API Response:{response.status_code} JSON Response {response.json()}")

url = "http://localhost:8000/predict_items_csv"
files = {'file': ('data.csv', open('data.csv', 'rb'))}
response = requests.post(url, files=files)
if response.status_code == 200:
    with open('response.csv', 'wb') as f:
        f.write(response.content)
    print(f"{url} API Response:{response.status_code}")
else:
    print(f"Error: {response.status_code}")

url = "http://localhost:8000/predict_items"
response = requests.post(url, headers=headers, json=data2)
print(f"{url} API Response:{response.status_code} JSON Response {response.json()}")