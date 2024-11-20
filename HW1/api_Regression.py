import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from io import StringIO
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib  # Для загрузки модели
import re
from sklearn.pipeline import Pipeline
import numpy as np
import tempfile

app = FastAPI()

model = joblib.load("model.pkl")  # Загружаем модель регрессии
scaler = joblib.load("scaler.pkl")  # Если использовалась стандартизация

def parse_torque(torque_str):
    if pd.isna(torque_str):
        return np.nan, np.nan
    torque_str = str(torque_str)
    torque_str = torque_str.lower().strip()
    if torque_str == 'null':
        return np.nan, np.nan
    value = None
    rpm_range = None
    units = None
    min_rpm_range = None
    #'210 / 1900'
    pattern = r'([\d\.]+)\s*(|nm)\s*(/)\s*([\d\,]+)\s*(|rpm)'
    match = re.match(pattern, torque_str)
    if match:
        value = float(match.group(1))
        if match.group(3) != '/':
          rpm_range = float(match.group(3))
        else:
          rpm_range = float(match.group(4))
        return value, rpm_range, min_rpm_range,1
    #'51nm@ 4000+/-500rpm'
    pattern = r'(\d+)\s*nm@?\s*(\d+)\s*\+/-\s*(\d+)\s*rpm'
    match = re.match(pattern, torque_str)
    if match:
        value = float(match.group(1))
        rpm_range = float(match.group(2))+float(match.group(3))
        min_rpm_range = float(match.group(2))-float(match.group(3))
        return value, rpm_range, min_rpm_range,2
    #48@ 3,000+/-500(nm@ rpm)
    pattern = r'(\d+)\s*@\s*([\d,]+)\s*\+/-\s*(\d+)\s*\(nm@ rpm\)'
    match = re.match(pattern, torque_str)
    if match:
        value = float(match.group(1).replace(',',''))
        rpm_range = float(match.group(2).replace(',',''))+float(match.group(3).replace(',',''))
        min_rpm_range = float(match.group(2).replace(',',''))-float(match.group(3).replace(',',''))
        return value, rpm_range, min_rpm_range,3
    pattern = r'([\d\.]+)\s*(|kgm|nm)\s*(@|at)\s*([\d\,]+)(-|~)([\d\,]+)\s*(|rpm)'
    match = re.match(pattern, torque_str)
    if match:
        if match.group(2) == 'kgm':
            value = float(match.group(1)) * 9.81  # конвертация в Nm
        value = float(match.group(1))
        rpm_range = float(match.group(6).replace(',',''))
        min_rpm_range = float(match.group(4).replace(',',''))
        return value, rpm_range, min_rpm_range,5
    #'20.4@ 1400-3400(kgm@ rpm)'
    pattern = r'([\d\.]+)\s*(@|at)\s*([\d\,]+)(-|~)([\d\,]+)(\()(kgm)\s*(@|at)\s*(rpm)(\))'
    match = re.match(pattern, torque_str)
    if match:
        if match.group(7) == 'kgm':
            value = float(match.group(1)) * 9.81  # конвертация в Nm
        value = float(match.group(1))
        rpm_range = float(match.group(5).replace(',',''))
        min_rpm_range = float(match.group(3).replace(',',''))
        return value, rpm_range, min_rpm_range,6
    pattern = r'([\d\.]+)\s*(kgm|nm)\s*(@|at)\s*([\d\,]+)\s*(|rpm)'
    match = re.match(pattern, torque_str)
    if match:
        value = float(match.group(1))
        rpm_range = float(match.group(4).replace(',',''))
        return value, rpm_range, min_rpm_range,4
    pattern = r'([\d\.]+)\s*(@|at)\s*([\d\,]+)(\()(kgm|nm)\s*(@|at)\s*(rpm)(\))'
    match = re.match(pattern, torque_str)
    if match:
        if match.group(5) == 'kgm':
            value = float(match.group(1)) * 9.81  # конвертация в Nm
        value = float(match.group(1))
        rpm_range = float(match.group(3).replace(',',''))
        return value, rpm_range, min_rpm_range,7
    pattern = r'(\d+)\s*nm\(([\d.]+)kgm\)@\s*(\d+)rpm'
    match = re.match(pattern, torque_str)
    if match:
        value = float(match.group(1))
        rpm_range = float(match.group(3).replace(',',''))
        return value, rpm_range, min_rpm_range,71
    pattern = r'(\d+)\s*nm'
    match = re.match(pattern, torque_str)
    if match:
        value = float(match.group(1))
        return value, rpm_range, min_rpm_range,8
    # Обработка возможных форматов

def extract_car_info(name):
    if 'sedan' in name.lower():
        return 'sedan'
    elif 'hatchback' in name.lower():
        return 'hatchback'
    elif 'suv' in name.lower() or 'muv' in name.lower():
        return 'suv'
    elif 'convertible' in name.lower():
        return 'convertible'
    else:
        return 'other'

def modify(df,scaler):
    df['mileage'] = df['mileage'].str.replace(r'[^0-9.]', '', regex=True)
    df['engine'] = df['engine'].str.replace(r'[^0-9.]', '', regex=True)
    df['max_power'] = df['max_power'].str.replace(r'[^0-9.]', '', regex=True)
    df['max_power'] = df['max_power'].replace('', np.nan)
    df['mileage'] = df['mileage'].astype(float)
    df['engine'] = df['engine'].astype(float)
    df['max_power'] = df['max_power'].astype(float)
    df['brand'] = df['name'].str.split(' ').str[0]
    df['car_type'] = df['name'].apply(extract_car_info)
    df['owner_is_third_or_more'] = df['owner'].apply(lambda x: 1 if 'third' in x.lower() or 'more' in x.lower() else 0)
    df['first_or_second_owner_and_dealer'] = df.apply(
        lambda row: 1 if ('First Owner' in row['owner'] or 'Second Owner' in row['owner']) and 'Dealer' in row[
            'seller_type'] else 0, axis=1)
    df[['torque_value', 'max_torque_rpm', 'min_rpm_range', 'torque_alg']] = df['torque'].apply(
        lambda x: pd.Series(parse_torque(x)))
    df = df.drop(columns=['name', 'selling_price', 'torque', 'torque_alg'])
    df = pd.get_dummies(df,
                        columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand',
                                 'car_type'], drop_first=True)
    df['power_per_liter'] = df['max_power'] / df['engine']

    df['year_squared'] = df['year'] ** 2
    df['year_engine'] = df['year'] * df['engine']
    df['max_torque_rpm'] = df['max_torque_rpm'].replace(np.nan, 0)
    df['min_rpm_range'] = df['min_rpm_range'].replace(np.nan, 0)

    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_

    columns_test = df.columns
    missing_columns = [col for col in feature_names if col not in columns_test]

    for col in missing_columns:
        df[col] = 0

    df2 = pd.DataFrame()

    for col in scaler.feature_names_in_:
        df2[col] = df[col]

    return df2

# Описание объекта автомобиля
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


# Описание коллекции объектов (список автомобилей)
class Items(BaseModel):
    objects: List[Item]

@app.post("/test")
def test_item(item: Item) -> float:
    print("It`s works")
    return 0

# Предсказание для одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> float:

    df = pd.DataFrame([item.dict()])
    df2 = modify(df,scaler)

    df_scaled = scaler.transform(df2)
    predicted_price = model.predict(df_scaled)

    return predicted_price


# Предсказание для коллекции объектов
@app.post("/predict_items_csv")
async def predict_items(file: UploadFile = File(...)):
    # Чтение данных из CSV
    contents = await file.read()

    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df_response = df.copy()
    df2 = modify(df, scaler)
    df_scaled = scaler.transform(df2)
    predicted_price = model.predict(df_scaled)
    df_response['predicted_price'] = model.predict(df_scaled)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_response.to_csv(temp_file.name, index=False)
    temp_file.close()

    print(predicted_price)
    return FileResponse(temp_file.name, media_type='text/csv', filename="predicted_data.csv")

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    items_list = [item.dict() for item in items.objects]

    df = pd.DataFrame(items_list)
    df2 = modify(df,scaler)

    df_scaled = scaler.transform(df2)
    predicted_price = model.predict(df_scaled)

    return predicted_price

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
