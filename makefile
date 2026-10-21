.venv:
	py -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\python -m pip install -r requirements.txt

build: .venv
	.venv\Scripts\python src\main.py

eda: .venv
	.venv\Scripts\python src\main.py --eda

diagnostics: .venv
	.venv\Scripts\python src\main.py --diagnostics

logistic: .venv
	.venv\Scripts\python src\main.py --logistic

predict: .venv
	.venv\Scripts\python src\main.py --predict

clean:
	rmdir /s /q .venv
