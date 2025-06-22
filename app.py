import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
from dotenv import dotenv_values
from qdrant_client import QdrantClient

env = dotenv_values(".env")

DATA = 'welcome_survey_simple_v2.csv'

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
)


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=";")
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters


with st.sidebar:
    st.header("Sprawdź do której grupy należysz!")
    st.markdown("Zobacz ile osób ma podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.radio("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.radio("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.radio("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    # Dodanie przycisku, aby wyświetlić wyniki
    show_results = st.button('Oczaruj mnie!')
    
    
    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

# Przed kliknięciem przycisku pokazujemy tekst zachęty
if not show_results:
    st.markdown(
    "<h1 style='font-size: 35px; text-align: center;'>🧙‍♂️ Tiara Przydziału 2 i ⅔ 🌀</h1>", 
    unsafe_allow_html=True
)
    st.markdown(
    "<h1 style='font-size: 30px; text-align: center;'>Hmmmmm.... Trudne, naprawdę trudne...</h1>", 
    unsafe_allow_html=True
)
    st.image("https://images.nightcafe.studio/jobs/wCkbew8LbK16eG9wOqvN/wCkbew8LbK16eG9wOqvN-O60Xk.jpeg?tr=w-1600,c-at_max")
    st.info("Wybierz swoje odpowiedzi i naciśnij przycisk 'Oczaruj mnie!', aby znaleźć grupę, do której najlepiej pasujesz!")


if show_results:
    model = get_model()
    all_df = get_all_participants()
    cluster_names_and_descriptions = get_cluster_names_and_descriptions()



    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]


    st.markdown(f"<h1 style='font-size: 40px;'>Najbliżej Ci do grupy {predicted_cluster_data['name']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 20px;'>{predicted_cluster_data['description']}</p>", unsafe_allow_html=True)
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

    if predicted_cluster_data['name'] == "Wodniacy z Wyższym Wykształceniem":
        st.image("https://cdn.galleries.smcloud.net/t/galleries/gf-9aJ8-JFgW-3h9U_surfujace-psy-664x442-nocrop.jpg")  # 0
    elif predicted_cluster_data['name'] == "Górscy Mędrcy":
        st.image("https://img.freepik.com/premium-zdjecie/ultra-realistyczny-medrzec-medytuje-w-krajobrazie_9493-8521.jpg")  # 1
    elif predicted_cluster_data['name'] == "Kociarze Górscy":
        st.image("https://i.wpimg.pl/1280x/m.fotoblogia.pl/snow-leopard-sascha-fone-6b2e0e5.jpg")  # 2
    elif predicted_cluster_data['name'] == "Leśni Filozofowie":
        st.image("https://debogora.com/data/include/img/news/1671551739.jpg")  # 3
    elif predicted_cluster_data['name'] == "Górscy Poszukiwacze Przygód":
        st.image("https://brubeck.pl/wp-content/uploads/2024/03/piesze-wedrowki-po-gorach2.webp")  # 4
    elif predicted_cluster_data['name'] == "Wodni Samotnicy":
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/0e/Jezioro_Wdzydze%2C_samotny_w%C4%99dkarz.jpg")  # 5
    elif predicted_cluster_data['name'] == "Wodni Entuzjaści":
        st.image("https://kitewyjazdy.pl/wp-content/uploads/2019/05/kite-surfing-at-Prasonisi-Rhodes.jpg")  # 6
    elif predicted_cluster_data['name'] == "Wodni Psiarze":
        st.image("https://cziko.com.pl/data/include/img/news/1680692347.jpg")  # 7
    elif predicted_cluster_data['name'] == "Leśni Młodzi Odkrywcy":
        st.image("https://bi.im-g.pl/im/3b/ce/13/z20769083IH,Dzieci-w-kanadyjskim-lesnym-przedszkolu-ForestKids.jpg")  # 8
    else:
        st.image("https://demotywatory.pl/uploads/202009/1600031304_9uzd9g_fb_plus.jpg")  # 9

    # Liczba wszystkich
    c0, c1, c2 = st.columns(3)
    with c0:
        st.metric("Liczba osób w grupie:", len(same_cluster_df))

    # Liczba mężczyzn
    with c1:
        st.metric("Liczba mężczyzn:", len(same_cluster_df[same_cluster_df["gender"] == "Mężczyzna"]))
    # Liczba kobiet
    with c2:
        st.metric("Liczba kobiet:", len(same_cluster_df[same_cluster_df["gender"] == "Kobieta"]))

    # Wykres wieku
    fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
    fig.update_layout(
        title={
            'text': "Rozkład wieku w grupie",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}
        },
        xaxis_title="",
        yaxis_title="Liczba osób",
        xaxis=dict(
            title_font=dict(size=25),  # Zwiększ czcionkę tytułu osi X
            tickfont=dict(size=18)  # Zwiększ czcionkę etykiet na osi X
        ),
        yaxis=dict(
            title_font=dict(size=25),  # Zwiększ czcionkę tytułu osi y
            tickfont=dict(size=18)  # Zwiększ czcionkę etykiet na osi y
        ),
    )

    st.plotly_chart(fig)


    # WYKRES KOŁOWY DLA WYKSZTAŁCENIA
    edu_level_counts = same_cluster_df["edu_level"].value_counts()
    fig = px.pie(values=edu_level_counts, names=edu_level_counts.index, title="Wykształcenie")
    color_map = {
        'Podstawowe': 'yellow',  # Kolor dla podstawowego wykształcenia
        'Średnie': 'indianred',   # Kolor dla średniego wykształcenia
        'Wyższe': 'dodgerblue'      # Kolor dla wyższego wykształcenia
    }
    fig.update_layout(
        title={
            'text': "Rozkład wykształcenia",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}
        },
        legend=dict(
            title="Wykształcenie",
            font=dict(size=18, color="black"),
            bgcolor="rgba(255, 255, 255, 1.5)",  # tło legendy
            bordercolor="black",  # kolor ramki
            borderwidth=1
        )
    )
    fig.update_traces(
        textinfo='percent+label',  # pokazuje etykiety i procenty
        textfont=dict(size=25, color='black'),  # większa czcionka na etykietach i procentach
        marker=dict(
            colors=[color_map.get(val, 'gray') for val in edu_level_counts.index]  # Kolory na podstawie wartości w edu_level
        )
    )
    st.plotly_chart(fig)



    # HISTOGRAM ULUBIONYCH ZWIERZĄT
    # Przypisanie kolorów na sztywno do poszczególnych kategorii
    color_map = {
        'Brak ulubionych': '#6E7F7F', # ołówkowy
        'Koty': '#FFD700',            # słonecznikowy
        'Psy': '#1E90FF',             # ciemny niebieski
        'Koty i Psy': '#8FB480',      # średnie wartości kolorów kota i psa
        'Inne': '#800000'             # Bordo
    }

    fig = px.histogram(same_cluster_df.sort_values("fav_animals"), x="fav_animals", 
                    color="fav_animals", color_discrete_map=color_map)

    fig.update_layout(
        title={
            'text': "Ulubione zwierzęta w grupie",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}
        },
        xaxis_title="",
        yaxis_title="Liczba osób",
        xaxis=dict(
            title_font=dict(size=25),  # Zwiększ czcionkę tytułu osi X
            tickfont=dict(size=18)  # Zwiększ czcionkę etykiet na osi X
        ),
        yaxis=dict(
            title_font=dict(size=25),  # Zwiększ czcionkę tytułu osi Y
            tickfont=dict(size=18)  # Zwiększ czcionkę etykiet na osi Y
        ),
        showlegend=False  # Ukryj legendę
    )

    # Wyświetlanie wykresu
    st.plotly_chart(fig)



    # HISTOGRAM DLA ULUBIONEGO MIEJSCA
    # Przypisanie kolorów na sztywno do poszczególnych kategorii
    color_map = {
        'Nad wodą': '#66B2FF',    # kolor spokojnego oceanu, jasny błękit, kojący i relaksujący :)
        'W lesie': '#2C6B2F',     # kolor lasu i przetrwania
        'W górach': '#6B4F31',    # górski brąz, siła gór
        'Inne': '#6E7F7F',        # ołówkowy
    }

    fig = px.histogram(same_cluster_df.sort_values("fav_place"), x="fav_place", 
                    color="fav_place", color_discrete_map=color_map)

    fig.update_layout(
        title={
            'text': "Ulubione miejsce",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}
        },
        xaxis_title="",
        yaxis_title="Liczba osób",
        xaxis=dict(
            title_font=dict(size=25),  # Zwiększ czcionkę tytułu osi X
            tickfont=dict(size=18)  # Zwiększ czcionkę etykiet na osi X
        ),
        yaxis=dict(
            title_font=dict(size=25),  # Zwiększ czcionkę tytułu osi Y
            tickfont=dict(size=18)  # Zwiększ czcionkę etykiet na osi Y
        ),
        showlegend=False  # Ukryj legendę
    )

    # Wyświetlanie wykresu
    st.plotly_chart(fig)


    # PŁEĆ
    # Liczba wystąpień płci
    gender_counts = same_cluster_df["gender"].value_counts()

    # Filtrujemy kategorie z zerową wartością
    gender_counts = gender_counts[gender_counts > 0]

    # Kolory dla płci
    color_map = {
        'Mężczyzna': '#1E3A5F',  # Kolor dla mężczyzn
        'Kobieta': '#F1A7C1',     # Kolor dla kobiet
    }

    # Tworzenie wykresu
    fig = px.pie(values=gender_counts, names=gender_counts.index, title="Rozkład płci")

    # Ustawienia wykresu
    fig.update_layout(
        title={
            'text': "Rozkład płci",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}
        },
        legend=dict(
            title="Płeć",
            font=dict(size=18, color="black"),
            bgcolor="rgba(255, 255, 255, 1.5)",
            bordercolor="black",
            borderwidth=1
        )
    )

    # Ustawienia dla wykresu
    fig.update_traces(
        textinfo='percent+label',
        textfont=dict(size=25, color='white'),
        marker=dict(
            colors=[color_map.get(val, 'gray') for val in gender_counts.index]
        )
    )

    # Wyświetlenie wykresu
    st.plotly_chart(fig)