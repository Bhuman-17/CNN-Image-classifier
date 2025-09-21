
# CIFAR-10 CNN – Image Classification

A compact convolutional neural network (CNN) project for classifying images from the CIFAR-10 dataset (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). This repo contains a Jupyter notebook for training/experimentation and a saved Keras model for quick inference.

---

## 📦 Repository Contents

- `Cnn_CIFAR_image_classification.ipynb` – main notebook to train/evaluate the CNN (open in Jupyter / VS Code).  
- `cifar10_model.h5` – pre-trained Keras model saved with `model.save()` for immediate use.  
- `requirements.txt` – Python dependencies used for this project.  
- `screenshot.png` – sample project cover/screenshot.  

> Tip: If you only want to run predictions, you can skip training and directly load `cifar10_model.h5`.

---

## 🛠️ Setup

### 1) Create & activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
````

If PowerShell blocks activation, run PowerShell **as Administrator** and set the execution policy once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then try activation again:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

**Windows (CMD):**

```bat
python -m venv .venv
.venv\\Scripts\\activate.bat
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Option A — Use the Notebook

Open `Cnn_CIFAR_image_classification.ipynb` and run all cells to train/evaluate the model. You can tweak architecture, epochs, and augmentation there.

### Option B — Quick Inference with the Saved Model

Use the pre-trained `cifar10_model.h5` to predict on your own images (32×32 RGB).

```python
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("cifar10_model.h5")

# CIFAR-10 class names
class_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((32, 32))
    x = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)  # (1, 32, 32, 3)

x = preprocess("your_image.jpg")
probs = model.predict(x)[0]
pred_idx = int(np.argmax(probs))
print("Prediction:", class_names[pred_idx], "– confidence:", float(probs[pred_idx]))
```

---

## 🧪 Expected Results

Depending on your exact architecture/epochs/augmentation, a simple CNN on CIFAR-10 often achieves **70–80%** test accuracy. With regularisation and augmentation, **80%+** is common; advanced architectures (ResNets, WideResNets) can go substantially higher.

---

## 🖼️ Screenshot

A cover image (`screenshot.png`) is included to help your README look good in the repo. Replace it later with your own training curves, confusion matrix, or sample predictions.

![Project Screenshot]<img width="773" height="826" alt="image" src="https://github.com/user-attachments/assets/ab4bb7cb-0eb4-4aad-adc4-ddbe0005c037" />


---

## ⚙️ Troubleshooting

* **Virtualenv activation error on Windows (PowerShell):**

  * Use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once), then re-run `.\.venv\Scripts\Activate.ps1`.
  * Alternatively, use **CMD** and activate with `.\.venv\Scripts\activate.bat`.

* **TensorFlow CPU vs GPU:**

  * The provided `requirements.txt` uses standard TensorFlow. For GPU acceleration, install the appropriate CUDA/cuDNN stack and `tensorflow[and-cuda]` for your platform/version.

* **Image shape errors:**

  * Ensure inputs are **(32, 32, 3)** and values are **scaled to \[0, 1]**.

---

## 📄 License

For personal/educational use. Replace with your preferred license if publishing.

---

## ✅ Verified Dependencies

See `requirements.txt` in this repo for tested versions.

```

---

Would you also like me to give you a ready-made **`predict.py`** file (outside Jupyter) so you can test your trained model on any image without opening the notebook?
```
