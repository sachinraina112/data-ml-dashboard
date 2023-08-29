import streamlit as st
import matplotlib
import pandas as pd
import time
import io
from PIL import Image
import joblib
# from configs.AppConfig import path_config
import uuid
import os
from dash import gen_uuid, training, all_plots, get_inference, create_output, preprocess


st.set_page_config(layout="wide")


@st.cache_data
def init():
    input_data_path = "../data/input/final.csv"
    out_prefix = "../data/output/"
    unique_id = gen_uuid()
    print(f"Unique folder {unique_id}")
    output_path = out_prefix + unique_id + "/"
    target = "popularity"
    df = pd.read_csv(input_data_path)
    proc = df.copy(deep=True)
    flag = True
    model = None
    if flag:
        create_output(output_path)
        train_pr = preprocess(proc)
        all_plots(train_pr, target, output_path)
    return input_data_path, output_path, target, df, train_pr

input_data_path, output_path, target, df, train_pr = init()

@st.cache_data
def run_inf(inf_input):
    path_to_model = "../models/model.pkl"
    mod = joblib.load(path_to_model)
    features = list(inf_input.keys())
    pred = get_inference(mod, inf_input, features)
    return pred[0]

@st.cache_data(max_entries=3)
def run_training(data, sel_hp, shap_flag, shap_type, num, output_path, save=True):
    print(f"Shap requested? {shap_flag}")
    if 0 < len(sel_hp) < 4 and shap_flag and shap_type:
        if shap_type != "summary":
            shap_params = {"type":shap_type, "fi":num}
        else:
            shap_params = {"type":shap_type, "fi":0}
        model, eval_metric, _ , features = training(data, sel_hp, output_path, save=True, shap=shap_flag ,shap_params=shap_params)
        return model, eval_metric, features
    return None, 0, []







   



first = st.container()
dataset = st.container()
analysis = st.container()
model_training = st.container()






with first:
    st.title("Data Science Machine Learning Project Dashboard")
    st.text("Predict popularity of music track from available data")

with dataset:
    st.header("Overview of dataset")
    st.text_area('Info about the features', '''
This is a sample dataset from Spotify API, tracks over a range of 113 different genres.Each track has some audio features associated with it explained below:

1. track_id: The ID for the track.
                 
2. track_name: Name of the track.
                 
3. popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of playsthe track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past.

4. duration_ms: The track length in milliseconds.
                 
5. explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown).
                 
6. danceability: Danceability describes how suitable a track is for dancing based on a combination of musicalelements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
                 
7. energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy,while a Bach prelude scores low on the scale.
                 
8. key: The key the track is in. Integers map to pitches using standard Pitch Class notation.E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
                 
9. loudness: The overall loudness of a track in decibels (dB).
                 
10. mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
                 
11. speechiness: Speechiness detects the presence of spoken words in a track.The more exclusively speech-like the recording (e.g. talk show, audio book, poetry)the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words.Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered,including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
                 
12. acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
                 
13. instrumentalness: Predicts whether a track contains no vocals. 'Ooh' and 'aah' sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly 'vocal'.The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.
                 
14. liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.A value above 0.8 provides strong likelihood that the track is live.
                 
15. valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
                 
16. tempo: The overall estimated tempo of a track in beats per minute (BPM).In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
                 
17. time_signature: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.
                 
18. track_genre: The genre in which the track belongs.
    ''')
    st.text("Overview of Dataset. Note name features (like artists) are not used in model")
    
    
    st.dataframe(df.iloc[::,2:], use_container_width=True)
    # df = st.text_input("Add Input text")
    
with analysis:
    st.header("Analysis")
    st.text("Stats of target: popularity")
    st.dataframe(df['popularity'].describe())
    st.text("Getting average popularity for a track_genre")
    pop_genre = df.groupby(["track_genre"]).agg({"popularity":"mean"}).reset_index()
    pop_genre_sorted = pop_genre.sort_values(["popularity"], ascending=False)
    st.dataframe(pop_genre_sorted, use_container_width=True)
    st.bar_chart(pop_genre_sorted[:10],
    y='popularity',
    x='track_genre',
    color='#ffaa0088'
)
    st.text("After Preprocessing: getting below correlation heatmap for all 16 features used and 1 target:")

    st.image(output_path + "corr.png")

    col1, col2 = st.columns(2)
    col1.header("So how is target distributed??")
    col1.image(output_path + "dist.png", use_column_width=True)

    # grayscale = original.convert('LA')
    col2.header("Deviation from normal distribution:")
    col2.image(output_path + "prob.png",use_column_width=True)

    st.header("Scatter Plots for features with target (popularity)")
    ll = [str(i) + ".png" for i in range(0,16)]
    c1, c2, c3, c4, c5 = st.columns((1, 1, 1, 1,1))
    

    c1.image(output_path + ll[0], use_column_width=True)
    c2.image(output_path + ll[1], use_column_width=True)
    c3.image(output_path + ll[2], use_column_width=True)
    c4.image(output_path + ll[3], use_column_width=True)
    c5.image(output_path + ll[4], use_column_width=True)

    c6, c7, c8, c9, c10 = st.columns((1, 1, 1, 1,1))
    c6.image(output_path + ll[5], use_column_width=True)
    c7.image(output_path + ll[6], use_column_width=True)
    c8.image(output_path + ll[7], use_column_width=True)
    c9.image(output_path + ll[8], use_column_width=True)
    c10.image(output_path + ll[9], use_column_width=True)

    c11, c12, c13, c14, c15 = st.columns((1, 1, 1, 1,1))
    c11.image(output_path + ll[10], use_column_width=True)
    c12.image(output_path + ll[11], use_column_width=True)
    c13.image(output_path + ll[12], use_column_width=True)
    c14.image(output_path + ll[13], use_column_width=True)
    c15.image(output_path + ll[14], use_column_width=True)

    st.subheader("Popularity Vs track_sent_score")
    st.text("No Major relationship as per scatter plot")
    # c16 = st.columns(1)
    st.image(output_path + ll[15], use_column_width=True)

with st.form("Model Training"):
    st.header("Enter Options for model training")
    train_flag = st.checkbox("Select for enabling training")
    selected_hp = st.multiselect(
    "Select maximum of three hyperparameters of choice for tuning:",['bootstrap','ccp_alpha','criterion',
    'max_depth','max_features','max_samples','min_samples_split', 'n_estimators','n_jobs'],
    max_selections=3,
)   
    print(f"Selected Hyper parameters {selected_hp}")
    shap_flag = st.checkbox("Select if SHAP plots are required")
    var_col, disp_col = st.columns(2)
    shap_type = var_col.selectbox("Select which SHAP plots are required", options=['summary','force',"both"], index=0)
    disp_col.text(f"Shap Type selected is {shap_type}")
    record_number = st.number_input('Insert a index of records to get force plot',0)
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        model, eval_metric, features = run_training(train_pr, selected_hp, shap_flag, shap_type, record_number, output_path, save=True)
        if model is not None:
            st.write(f"Evaluation Metric RMSE is {round(eval_metric, 3)}")
        if shap_flag:    
            col1, col2 = st.columns(2)
            if shap_type != "force":
                col1.header("SHAP Influencing Factors")
                col1.image(output_path + "shap_summary_bar.png", use_column_width=True)
            if shap_type != "summary":
                col2.header("SHAP Force plot for individual record")
                col2.image(output_path + "shap_force.png",use_column_width=True)
                
                    

with st.form("Inference"):
    st.header("Enter for Inference")
    inference_flag = st.checkbox("Select for inference")
    
    duration_ms = st.slider("duration_ms",1,1000000)
    explicit = st.slider("explicit",0,1,step=1)
    danceability = st.slider("danceability",0.0,1.0, step=0.2)
    energy = st.slider("energy",0.0,1.0, step=0.2)
    key = st.slider("key",0,15, step=5)
    loudness = st.slider("loudness",-50,50, step=10)
    mode = st.slider("mode",0.0,1.0, step=0.2)
    speechiness = st.slider("speechiness",0.0,1.0, step=0.2)
    acousticness = st.slider("acousticness",0.0,1.0, step=0.2)
    instrumentalness = st.slider("instrumentalness",0.0,1.0, step=0.2)
    liveness = st.slider("liveness",0.0,1.0, step=0.2)
    valence = st.slider("valence",0.0,1.0, step=0.2)
    tempo = st.slider("tempo",0,250, step=50)
    time_signature = st.slider("time_signature",0,5, step=1)
    track_genre = st.slider("track_genre",0,113, step=1)
    track_name_sent_score = st.slider("track_name_sent_score",-1.0,1.0, step=0.2)
    submitted_inf = st.form_submit_button("Submit")


    
    st.header("Submit for running inference on sample record ......")
        
    if submitted_inf:
        st.text("Submitted Inference")
        inf_input = {"duration_ms":duration_ms, "explicit":explicit, "danceability":danceability, 
                            "energy": energy, "key":key, "loudness":loudness, "mode":mode, 
                            "speechiness":speechiness, "acousticness":acousticness, 
                            "instrumentalness":instrumentalness, "liveness":liveness,
                            "valence":valence, "tempo":tempo, "time_signature":time_signature, 
                            "track_genre": track_genre, "track_name_sent_score":track_name_sent_score}
        pred = run_inf(inf_input)
        st.write(f"Prediction popularity is {pred}")
        st.stop()


    
   








# with sixth:
#     matplotlib.pyplot.close()
#     var_col, disp_col = st.columns(2)
#     intensity = var_col.selectbox("Select any three hyperparameters for tuning", options=['low','high'], index=0)
#     blend_type = var_col.selectbox("Select if content and input images are same or not for style intensity", options=['only-style','style-content','new-blend'], index=0)
#     # 