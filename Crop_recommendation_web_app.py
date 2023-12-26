import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model=pickle.load(open('Crop_model.sav','rb'))

img = '''
<style>
.stApp {
    background-image: url("https://th.bing.com/th/id/OIP.V-msygM0OeeV9Qkg6qFU-wHaFj?pid=ImgDet&w=200&h=150&c=7&dpr=1.3");
    background-size: cover;
    background-position: top center;
    background-repeat: no-repeat;
    background-attachment: local;
    opacity:1;
}
</style>
'''
st.markdown(img, unsafe_allow_html=True)

#creating a function for prediction

def crop_prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_numpy_array=np.array(input_data,dtype=float)
    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)

    # Mapping predicted label to crop
    crop_mapping = {0: 'Apple',1: 'Banana',2: 'Blackgram',3: 'Chickpea',4: 'Coffee',5: 'Coconut',6: 'Cotton',7: 'Grapes',
                    8: 'Jute',9: 'Kidneybeans',10: 'Lentil',11: 'Maize',12: 'Mango',13: 'Mothbeans',14: 'Mungbean',
                    15: 'Muskmelon',16: 'Orange',17: 'Papaya',18: 'Pigeonpeas',19: 'Pomegranate',20: 'Rice',21: 'Watermelon'
                   }
    # Get the crop name based on the predicted label
    predicted_crop = crop_mapping[prediction[0]]
    return "The recommended crop is: {}".format(predicted_crop)



def main():
    #page title
    st.title("Crop Recommendation Using ML")

    N=st.text_input("Enter the ratio of Nitrogen content in soil in kg/ha")

    P=st.text_input("Enter the ratio of Phosporous content in soil in kg/ha")

    K=st.text_input("Enter the ratio of Pottasium content in soil in kg/ha ")

    temperature=st.text_input("Enter the temperature in degree Celsius")

    humidity=st.text_input("Enter the  relative humidity in percentage")

    ph=st.text_input("Enter the ph value of the soil")

    rainfall=st.text_input("Enter the rainfall in mm")

    #code for prediction

    Report=''

    #creating a button for prediction

    if st.button('Crop Recommendation Result'):
        Report=crop_prediction([N,P,K,temperature,humidity,ph,rainfall])

    st.success(Report)


if __name__ =='__main__':
    main()
    


