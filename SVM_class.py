#SVM Model Trainer for Botanical Classification
#@R.Maia
# 05/01/2024

import rasterio
import rasterio.mask
import joblib
import numpy as np
import geopandas as gpd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Paths to the folders containing the TIFF images and shapefiles
caminho_pasta_tiff = (r'C:\Users\raster')
caminho_pasta_shape = (r'C:\Users\shapes')

# Lists all TIFF and shapefile files
arquivos_tiff = [f for f in os.listdir(caminho_pasta_tiff) if f.endswith('.tif')]
arquivos_shape = [f for f in os.listdir(caminho_pasta_shape) if f.endswith('.shp')]

# Make sure the number of TIFF and shapefile files is equal
assert len(arquivos_tiff) == len(arquivos_shape), "Uneven number of TIFF and shapefile files."

# Processes each pair of files
for arquivo_tiff, arquivo_shape in zip(arquivos_tiff, arquivos_shape):
    caminho_tiff = os.path.join(caminho_pasta_tiff, arquivo_tiff)
    caminho_shape = os.path.join(caminho_pasta_shape, arquivo_shape)

    # Upload the shapefile
    gdf = gpd.read_file(caminho_shape)

    # Upload the image and extract features and labels
    X = []  # Features
    y = []  # labels

    with rasterio.open(caminho_tiff) as src:
        for _, row in gdf.iterrows():
            out_image, out_transform = rasterio.mask.mask(src, [row['geometry']], crop=True)
            if out_image.size == 0:
                continue

            n_bands, n_rows, n_cols = out_image.shape
            out_image_reshaped = out_image.reshape(n_bands, n_rows * n_cols).T
            X.extend(out_image_reshaped)
            y.extend([row['id']] * out_image_reshaped.shape[0])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the SVM model
    model = SVC()

    # Search space for hyperparameters
    param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.1, 10], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

    # Create the GridSearchCV
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3)

    # Train the model
    grid.fit(X_train, y_train)

    # Print the best parameters
    print(f"Best Parameters for{arquivo_tiff}: {grid.best_params_}")

    # Forecasts & Evaluation
    y_pred = grid.predict(X_test)
    print(f"Accuracy for {arquivo_tiff}: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Save the model
    nome_modelo = f'model_svm_{arquivo_tiff.replace(".tif", "")}.pkl'
    joblib.dump(grid.best_estimator_, nome_modelo)
