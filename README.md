# evlongsim

## Scope:
4 Wheel drive model of RC Car. This will be used to spec a battery and motor for FRC. Will try to incorporate as many nonlinearities as possible with the motor, battery, and controller. This script will simulate a straightline acceleration at the limits of the battery, tires, motor, and aerodynamics. Drag and Lift is considered as estimations. Bluff body drag will be considered

### How to start up sim
1. (Optional) Create and activate a virtual python environment
2. Install the requirements in the requirements file (python3 -m pip install -r requirements.txt) (remove the 3 for windows)
3. Either in vscode terminal or your OS terminal, navigate to the directory and start the main.py file with streamlit (streamlit run main.py)
4. (Possibly Neccesary) Install reccommended streamlit dependencies

### How to run (This will change as the app matures)
1. In your code editor change vehicle, motor, battery, and tire parameters as you see fit (This will change to sliders and inputs in the streamlit browser)
2. The output data is placed into a dataframe. Plot different channels as you would like
