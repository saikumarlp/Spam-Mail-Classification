import streamlit as st
import pickle
import sklearn
model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vec.pkl','rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.. ")
    st.subheader("Classification")
    user_input = st.text_area("Enter an Email to Classify ",height=150)
    if st.button("Classify"):
        if user_input:
            data = [user_input]
            print(data)
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == "ham":
                st.success("This is Not A Spam Email")
            else:
                st.error("Caution!!!" "This is A Spam Mail")
        else:
            st.write("Please Enter an Email to Classify..")

main()

