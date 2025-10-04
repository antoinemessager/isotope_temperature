# Paleotemperature Reconstruction from Isotope Profiles

This project implements an inverse model based on the work of Passey et al. (2005) to reconstruct seasonal temperature profiles from isotopic measurements (δ¹⁸O) taken on horse teeth.

## Project Goal

Isotopic measurements made along a growing tooth (e.g., from a horse) do not represent an instantaneous temperature but rather a time-averaged signal due to the enamel mineralization process.

The objective of this code is to:
1.  Use an **inverse model** to "deconvolve" the averaged isotopic signal and estimate the original seasonal temperature signal.
2.  Fit a **sinusoidal model** to this reconstructed temperature profile to visualize and analyze seasonality (amplitude, mean, phase).
3.  Apply this analysis automatically to a dataset of multiple specimens (horses).

## Project Structure

The project is organized as follows:

```
.
├── data/
│   ├── equus_alldata.csv   # Raw data
│   └── results/            # Directory where PDF plots are saved
├── src/
│   ├── passey_model_core.py    # Core functions of the inverse model
│   └── curve_fitting_tools.py  # Tools for curve fitting and visualization
├── demo.ipynb              # Jupyter notebook for demonstration and analysis
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```

## Installation

Follow these steps to set up your environment and run the project.

**1. Clone the Repository**
```bash
git clone <URL_of_your_repository>
cd <project_folder_name>
```

**2. Create a Virtual Environment**
It is highly recommended to use a virtual environment to isolate the project's dependencies.
```bash
python -m venv venv
```

**3. Activate the Virtual Environment**
* On macOS / Linux:
    ```bash
    source venv/bin/activate
    ```
* On Windows:
    ```bash
    .\venv\Scripts\activate
    ```

**4. Install Dependencies**
Install all required libraries listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Usage

The main analysis is performed in the `demo.ipynb` Jupyter Notebook.

**1. Launch Jupyter**
Make sure your virtual environment is activated, then launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab
# or
jupyter notebook
```

**2. Run the Notebook**
Open the `demo.ipynb` file in your browser and run the cells in order.

## Analysis Workflow (as described in `demo.ipynb`)

The notebook follows these steps:
1.  **Data Loading**: The raw data is read from `data/equus_alldata.csv`.
2.  **Preprocessing**: The data is cleaned, and the δ¹⁸O values are converted into temperature estimates.
3.  **Analysis of a Single Specimen**: The notebook details the application of the inverse model and seasonal fitting on a single horse as an example. This allows for the visualization of intermediate results.
4.  **Batch Processing**: A loop iterates through all unique horses in the dataset, applies the complete analysis pipeline, and saves a plot of the results as a PDF file in the `data/results/` directory.

---
*Main Reference: Passey, B. H., et al. (2005). Inverse methods for estimating primary input signals from time-averaged isotope profiles. Geochimica et Cosmochimica Acta, 69(16), 4101-4116.*