# Handwritten Digit Classifier

A lightweight and user-friendly web application that identifies
handwritten digits (0--9) using a trained Artificial Neural Network
(ANN). Users can upload an image and instantly view the predicted digit.

🔗 **Live App:**\
https://digitrecognizer-h9eqfpr22kaei75lmrnfbk.streamlit.app/

## 🚀 Features

- Upload handwritten digit images (`png`, `jpg`, `jpeg`)
- Automatic preprocessing with OpenCV + NumPy
- ANN-based prediction
- Clean and simple Streamlit interface
- Instant results without any setup required

## 📁 Project Structure

    .
    ├── app.py                 # Streamlit app file
    ├── model.h5               # Trained ANN model
    ├── requirements.txt       # Python dependencies
    ├── app.ipynb              # Development notebook
    └── README.md              # Project documentation

## 🧠 How It Works

1.  User uploads an image through Streamlit.
2.  Image is read using PIL and converted to OpenCV array.
3.  Converted to grayscale, resized to 28x28 pixels, and normalized.
4.  ANN model predicts the digit.
5.  Predicted result is displayed.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-link>
cd <project-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## 📦 Dependencies

    streamlit
    tensorflow
    numpy
    opencv-python
    Pillow

## ☁️ Deployment (Streamlit Community Cloud)

1.  Push your repository to GitHub.
2.  Visit https://share.streamlit.io/.
3.  Connect your repository.
4.  Select the entry point file (`app.py`).
5.  Deploy.

## 📘 Model Information

- Type: Artificial Neural Network
- Dataset: MNIST (or equivalent)
- Input shape: 28x28 grayscale
- Output: Digit class (0--9)

## 🔮 Future Enhancements

- Add drawing canvas
- Show prediction confidence
- Detect multiple digits
- Improve UI design
