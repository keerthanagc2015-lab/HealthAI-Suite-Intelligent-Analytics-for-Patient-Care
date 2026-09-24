import os, sys, importlib.util
from datetime import date, timedelta
import requests
import streamlit as st

st.set_page_config(page_title='HealthAI Virtual Hospital', page_icon='🏥', layout='wide')
API='http://127.0.0.1:8000'
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
AGENT=os.path.join(ROOT,'src','17_agentic_ai')

st.markdown('''<style>
.hero{padding:28px;border-radius:18px;background:linear-gradient(135deg,#eaf4ff,#f3fbff,#eefaf5);border:1px solid #dce8f4;margin-bottom:20px}.hero h1{font-size:42px;margin:0 0 8px}.hero p{font-size:18px;color:#526274}.card{padding:20px;border:1px solid #e1e7ef;border-radius:16px;background:#fff;margin-bottom:12px}.muted{color:#6b7785;font-size:13px}
</style>''', unsafe_allow_html=True)

def api(path,payload=None,method='POST'):
    try:
        r=requests.get(API+path,timeout=120) if method=='GET' else requests.post(API+path,json=payload,timeout=120)
        try:d=r.json()
        except:d={'raw':r.text}
        return r.status_code,d
    except Exception as e:return None,{'error':str(e)}

def show(code,data):
    if code==200: st.success('Completed successfully.'); st.json(data)
    else: st.error(data.get('error',f'HTTP {code}')); st.json(data)

@st.cache_resource
def agent_module():
    p=os.path.join(AGENT,'03_agent_executor.py'); spec=importlib.util.spec_from_file_location('agent_exec',p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run_agent(q):
    m=agent_module()
    if hasattr(m,'run_agent'): return m.run_agent(query=q)
    return m.execute_agent_query(query=q)

if 'appointments' not in st.session_state: st.session_state.appointments=[]
if 'chat' not in st.session_state: st.session_state.chat=[]

st.sidebar.title('🏥 HealthAI')
portal=st.sidebar.radio('Portal',['🏠 Home','👨‍⚕️ Doctor Portal','👤 Patient Portal'])

# HOME
if portal=='🏠 Home':
    st.markdown('<div class="hero"><h1>🏥 HealthAI Virtual Hospital</h1><p>Intelligent healthcare platform combining Machine Learning, Deep Learning, Medical NLP, RAG and Agentic AI.</p></div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric('AI Tools','10'); b.metric('RAG Sources','8'); c.metric('ML / DL Models','Multiple'); d.metric('Live Visitors', 'DEMO')
    st.caption('Live Visitors is a simulated portfolio/demo metric until a real analytics backend is connected.')
    st.divider(); st.subheader('Choose your portal')
    x,y=st.columns(2)
    with x: st.markdown('<div class="card"><h2>👨‍⚕️ Doctor Portal</h2><p>🫁 Chest X-ray · 🏨 Length of Stay · 🧬 Primary Diagnosis · 💬 Patient Sentiment · 🧠 Medical NER</p></div>',unsafe_allow_html=True)
    with y: st.markdown('<div class="card"><h2>👤 Patient Portal</h2><p>🤖 Agentic AI · ❤️ Health Risk · 📅 Appointments · 📚 Medical Information · 🏥 Hospital Services</p></div>',unsafe_allow_html=True)
    st.subheader('Hospital website features')
    for icon,name,desc in [('👨‍⚕️','Find a Doctor','Departments and doctor discovery'),('📅','Appointments','Digital appointment requests'),('🧪','Laboratory','Lab services and reports'),('🫁','Radiology','Imaging services'),('💊','Pharmacy','Medication support'),('🚑','Emergency','Emergency information'),('📋','Health Records','Digital record integration'),('💳','Insurance & Billing','Coverage and billing information'),('📞','Contact','Hospital contact and support'),('📚','Health Library','Trusted health education')]:
        with st.container(border=True): st.write(f'{icon} **{name}** — {desc}')

# DOCTOR
elif portal=='👨‍⚕️ Doctor Portal':
    page=st.sidebar.radio('Doctor Services',['Dashboard','🫁 Chest X-ray','🏨 Length of Stay','🧬 Primary Diagnosis','💬 Patient Sentiment','🧠 Medical NER'])
    if page=='Dashboard':
        st.title('👨‍⚕️ Doctor Dashboard'); st.caption('Clinical intelligence and decision-support workspace')
        a,b,c,d=st.columns(4); a.metric('Patients Today','128'); b.metric('Pending Reviews','14'); c.metric('X-rays','37'); d.metric('Appointments','26')
        st.divider(); st.info('Use the sidebar to open each clinical AI tool.')
    elif page=='🫁 Chest X-ray':
        st.title('🫁 Chest X-ray Analysis'); f=st.file_uploader('Upload chest X-ray',type=['png','jpg','jpeg'])
        if f:
            st.image(f,use_container_width=True); st.warning('The CNN/adapter is implemented. The current FastAPI layer does not yet expose an image endpoint, so this screen is ready for the X-ray API bridge.')
    elif page=='🏨 Length of Stay':
        st.title('🏨 Hospital Length-of-Stay Prediction'); a,b=st.columns(2)
        with a:r=st.number_input('Diagnosis Count',0,20,3); g=st.selectbox('Gender',['M','F']); h=st.number_input('Hematocrit',value=40.0); n=st.number_input('Neutrophils',value=65.0); s=st.number_input('Sodium',value=139.0); gl=st.number_input('Glucose',value=110.0)
        with b:cr=st.number_input('Creatinine',value=1.0); bmi=st.number_input('BMI',value=25.0); pulse=st.number_input('Pulse',value=78.0); resp=st.number_input('Respiration',value=18.0); sec=st.number_input('Secondary Diagnosis Count',0,20,1); fac=st.text_input('Facility','A')
        if st.button('Predict Length of Stay',type='primary'):
            p={'patient_data':{'rcount':r,'gender':g,'dialysisrenalendstage':0,'asthma':0,'irondef':0,'pneum':0,'substancedependence':0,'psychologicaldisordermajor':0,'depress':0,'psychother':0,'fibrosisandother':0,'malnutrition':0,'hemo':0,'hematocrit':h,'neutrophils':n,'sodium':s,'glucose':gl,'bloodureanitro':15.0,'creatinine':cr,'bmi':bmi,'pulse':pulse,'respiration':resp,'secondarydiagnosisnonicd9':sec,'facid':fac}}
            show(*api('/predict/los',p))
    elif page=='🧬 Primary Diagnosis':
        st.title('🧬 Primary Diagnosis Prediction'); a,b=st.columns(2)
        with a:age=st.number_input('Age',1,120,52); gender=st.selectbox('Gender',['Male','Female']); region=st.text_input('Region','Tamil Nadu'); socio=st.selectbox('Socioeconomic Status',['Low','Middle','High']); symptoms=st.text_area('Symptoms','frequent urination, excessive thirst')
        with b:gl=st.number_input('Blood Glucose (mg/dL)',value=165.0); hb=st.number_input('HbA1c (%)',value=7.8); ch=st.number_input('Total Cholesterol (mg/dL)',value=220.0); bmi=st.number_input('BMI',value=29.5)
        if st.button('Predict Primary Diagnosis',type='primary'): show(*api('/predict/diagnosis',{'Age':age,'Gender':gender,'Region':region,'Socioeconomic_Status':socio,'Symptoms':symptoms,'Blood_Glucose_mg_dL':gl,'HbA1c_%':hb,'Total_Cholesterol_mg_dL':ch,'BMI':bmi}))
    elif page=='💬 Patient Sentiment':
        st.title('💬 Patient Sentiment'); t=st.text_area('Patient feedback','The staff were helpful and explained everything clearly.')
        if st.button('Analyze Sentiment',type='primary'): show(*api('/predict/sentiment',{'text':t}))
    elif page=='🧠 Medical NER':
        st.title('🧠 Medical NER'); t=st.text_area('Clinical note','Patient has critical limb ischaemia with stump pain. Started metformin 500mg bd and apixaban.',height=180)
        if st.button('Extract Medical Entities',type='primary'): show(*api('/predict/ner',{'text':t}))

# PATIENT
else:
    page=st.sidebar.radio('Patient Services',['Dashboard','❤️ Health Risk','🤖 HealthAI Assistant','📅 Appointments','🏥 Hospital Services'])
    if page=='Dashboard':
        st.title('👤 Patient Dashboard'); a,b,c,d=st.columns(4); a.metric('Appointments',len(st.session_state.appointments)); b.metric('Health Assessments','1'); c.metric('AI Assistant','Available'); d.metric('Digital Services','24/7'); st.info('Use the sidebar to access your health assessment, assistant and appointments.')
    elif page=='❤️ Health Risk':
        st.title('❤️ Personal Health Risk Assessment'); a,b=st.columns(2)
        with a:age=st.number_input('Age',1,120,30); gender=st.selectbox('Gender',['Male','Female']); gl=st.number_input('Blood Glucose (mg/dL)',50.0,500.0,100.0); hb=st.number_input('HbA1c (%)',2.0,20.0,5.5)
        with b:ch=st.number_input('Total Cholesterol (mg/dL)',50.0,500.0,180.0); bmi=st.number_input('BMI',10.0,70.0,24.0); region=st.text_input('Region','Tamil Nadu'); socio=st.selectbox('Socioeconomic Status',['Low','Middle','High'])
        if st.button('Assess My Risk',type='primary'):
            p={'patient_data':{'Age':age,'Blood_Glucose_mg_dL':gl,'HbA1c_%':hb,'Total_Cholesterol_mg_dL':ch,'BMI':bmi,'Gender':gender,'Region':region,'Socioeconomic_Status':socio,'Symptoms':'','BMI_Category':'Overweight' if bmi>=25 else 'Normal','Age_Group':'Middle Age' if 40<=age<60 else 'Other','High_Glucose':1 if gl>=140 else 0,'High_Cholesterol':1 if ch>=200 else 0,'HbA1c_Category':'High' if hb>=6.5 else 'Normal'}}
            code,res=api('/predict/diabetes',p)
            if code==200:
                prob=float(res.get('diabetes_probability',0)); st.metric('Estimated diabetes probability',f'{prob:.1%}'); st.progress(min(max(prob,0),1)); st.warning('This is a machine-learning risk estimate, not a medical diagnosis.')
            else:show(code,res)
    elif page=='🤖 HealthAI Assistant':
        st.title('🤖 HealthAI Assistant'); st.caption('Agentic AI selects the appropriate HealthAI capability.')
        for m in st.session_state.chat:
            with st.chat_message(m['role']): st.write(m['text'])
        q=st.chat_input('Ask a healthcare question...')
        if q:
            st.session_state.chat.append({'role':'user','text':q}); st.chat_message('user').write(q)
            with st.chat_message('assistant'):
                with st.spinner('HealthAI is routing your request...'): res=run_agent(q)
                if res.get('status')=='error': st.error(res.get('error','Agent failed.'))
                else: st.write(res.get('answer',res.get('message','HealthAI completed the request.'))); st.expander('Agent details').json(res)
    elif page=='📅 Appointments':
        st.title('📅 Appointments'); a,b=st.columns(2)
        with a:name=st.text_input('Patient name'); dept=st.selectbox('Department',['General Medicine','Cardiology','Endocrinology','Gynecology','Pediatrics','Radiology','Neurology'])
        with b:dt=st.date_input('Preferred date',min_value=date.today(),value=date.today()+timedelta(days=1)); tm=st.selectbox('Preferred time',['09:00 AM','10:00 AM','11:00 AM','02:00 PM','03:00 PM','04:00 PM'])
        reason=st.text_area('Reason for appointment')
        if st.button('Request Appointment',type='primary'):
            if not name.strip():st.error('Please enter patient name.')
            else:st.session_state.appointments.append({'patient':name,'department':dept,'date':str(dt),'time':tm,'reason':reason,'status':'Requested'}); st.success('Appointment request submitted.')
        st.divider(); st.subheader('My requests')
        for ap in st.session_state.appointments: st.write(f"**{ap['department']}** — {ap['date']} {ap['time']} — {ap['status']} — {ap['patient']}")
    else:
        st.title('🏥 Hospital Services')
        for icon,name,desc in [('👨‍⚕️','Find a Doctor','Doctor and department discovery'),('📅','Appointments','Book/request appointments'),('🧪','Laboratory','Laboratory services'),('🫁','Radiology','Imaging services'),('💊','Pharmacy','Medication support'),('🚑','Emergency','Emergency-care information'),('📋','Health Records','Digital records integration'),('💳','Insurance & Billing','Coverage and billing'),('📞','Contact Hospital','Support and contact information'),('📚','Health Library','Trusted health education')]: st.markdown(f'<div class="card"><h3>{icon} {name}</h3><p>{desc}</p></div>',unsafe_allow_html=True)

st.divider(); st.caption('HealthAI Virtual Hospital • Portfolio / Demonstration Platform • AI outputs do not replace professional medical care.')
