# SVM Model Trainer for Botanical Classification

This Python script is designed to train Support Vector Machine (SVM) models for botanical classification using geospatial data, specifically TIFF images and shapefiles. It utilizes the scikit-learn library for model implementation and evaluation.

## Author
- **R. Maia**

## Date
- Created: 05/01/2024

## Usage
1. Ensure you have Python 3.x installed on your system.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Update the paths to the folders containing the TIFF images and shapefiles in the script (`caminho_pasta_tiff` and `caminho_pasta_shape` variables).
4. Run the script `svm_model_trainer.py`.
5. The script will process each pair of TIFF and shapefile files, extract features and labels, train SVM models with hyperparameter tuning using GridSearchCV, evaluate model performance, and save the trained models to disk.

## Requirements
- Python 3.x
- scikit-learn
- rasterio
- geopandas

## File Structure
- `svm_model_trainer.py`: Main Python script for training SVM models.
- `README.md`: This README file providing an overview of the project.
- `requirements.txt`: File containing the list of required Python libraries.

