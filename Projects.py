import streamlit as st
import pandas as pd
import pickle

st.write("""

## TCAS
Let's enjoy **data science** project!T.T

""")

st.sidebar.header('TCAS')
st.sidebar.subheader('Please enter your data:')

def get_input():
    t_acY = st.sidebar.selectbox('AcademicYear',['','2561','2562','2563','2564','2565'])
    t_Sex = st.sidebar.radio('Sex', ['Male','Female'])
    t_fid = st.sidebar.text_input('FacultyID')
    t_dc = st.sidebar.text_input('DepartmentCode')
    t_etid = st.sidebar.text_input('EntryTypeID')
    t_egid = st.sidebar.text_input('EntryGroupID')
    t_tcas = st.sidebar.selectbox('TCAS',['','1','2','3','4','5'])
    t_egpa =st.sidebar.text_input('EntryGPA')
    t_reg = st.sidebar.selectbox('HomeRegion',['','International','North','South','West','East','North East','North West','South East','South West','Central'])
    t_stut = st.sidebar.selectbox('StudentTH',['','Yes','No'])
    t_coun = st.sidebar.text_input('Country(If you is Foreign)')
    t_srn_eng=st.sidebar.selectbox('SchoolRegionName(Eng)',['Foreign','Northern','Northeast','Central','Southern','Eastern','Western'])
    t_rn_religi = st.sidebar.selectbox('ReligionName',['Buddha','Christ','Islamic',])
    t_gpax =st.sidebar.text_input('GPAX')
    t_gpae =st.sidebar.text_input('GPA_Eng')
    t_gpam =st.sidebar.text_input('GPA_Math')
    t_gpas =st.sidebar.text_input('GPA_Sci')
    t_gpaso =st.sidebar.text_input('GPA_Sco')
    t_sta =st.sidebar.text_input('Status')
    
#Sex
    if t_Sex == 'Male':
        t_Sex = 'M'
    else:
        t_Sex = 'F'


    data = {
            'AcademicYear':t_acY,
            'Sex': t_Sex,
            'FacultyID':t_fid,
            'DepartmentCode':t_dc,
            'EntryTypeID':t_etid,
            'EntryGroupID':t_egid,
            'TCAS':t_tcas,
            'EntryGPA':t_egpa,
            'HomeRegion':t_reg,
            'StudentTH':t_stut,
            'Country(If you is Foreign)':t_coun,
            'SchoolRegionName(Eng)':t_srn_eng ,
            'ReligionName':t_rn_religi,
            'GPAX':t_gpax,
            'GPA_Eng':t_gpae,
            'GPA_Math' : t_gpam,
            'GPA_Sci':t_gpas,
            'GPA_Sco':t_gpaso,
            'Status':t_sta,

            }

    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.header('Application of TCAS\'s :')
st.subheader('TCAS:')
st.write(df)
        
data_sample = pd.read_excel ('tcas.xlsx')
df = pd.concat([df, data_sample],axis=0)

cat_data = pd.get_dummies(df[['AcademicYear', 'Sex', 'FacultyID', 'DepartmentCode', 'EntryTypeID',
       'EntryGroupID', 'TCAS', 'EntryGPA', 'HomeRegion', 'StudentTH',
       'Country', 'SchoolRegionNameEng', 'ReligionName', 'GPAX', 'GPA_Eng',
       'GPA_Math', 'GPA_Sci', 'GPA_Sco', 'Status']])

X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] 

X_new = X_new.drop(columns=['AcademicYear', 'Sex', 'FacultyID', 'DepartmentCode', 'EntryTypeID',
       'EntryGroupID', 'TCAS', 'EntryGPA', 'HomeRegion', 'StudentTH',
       'Country', 'SchoolRegionNameEng', 'ReligionName', 'GPAX', 'GPA_Eng',
       'GPA_Math', 'GPA_Sci', 'GPA_Sco', 'Status'])

st.subheader('Pre-Processed Input:')
st.write(X_new)

load_sc = pickle.load(open('normalization.pkl', 'rb'))

X_new = load_sc.transform(X_new)

st.subheader('Normalized Input:')
st.write(X_new)

load_knn = pickle.load(open('best_knn.pkl', 'rb'))

prediction = load_knn.predict(X_new)

st.subheader('Prediction:')
st.write(prediction)
    