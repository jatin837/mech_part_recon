dat: assets/blnw-images-224.zip
	mkdir dat/ && unzip assets/blnw-images-224.zip -d dat/
	
run: main.py
	./env/bin/python main.py
