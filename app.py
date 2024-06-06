import streamlit as st
from Funciones import plot_chart, descargar_datos, cargar_limpiar_datos, crear_tabla_por_dia, crear_tabla_por_horas, predict_values, plot_chart_pred
import joblib

def main():

    st.set_page_config(page_title='xxxxxxx', layout='wide',
                #   initial_sidebar_state=st.session_state.get('sidebar_state', 'collapsed'),
    )

    st.title('Precio del Bitcoin')

    st.write("### Text here")

    # Interactive widgets
    st.sidebar.header('Controls')
    horas_pred = st.sidebar.slider('Horas que desea predecir', min_value=1, max_value=100, value=30, step=1)

    url = descargar_datos()

    df = cargar_limpiar_datos(url)

    df_horas = crear_tabla_por_horas(df)

    df_dias = crear_tabla_por_dia(df)

    plt_horas = plot_chart(df_horas,'BTC/USDT precio por horas')

    plt_dias = plot_chart(df_dias,'BTC/USDT precia por dias')
    
    st.plotly_chart(plt_horas)

    st.plotly_chart(plt_dias)

    model = joblib.load('model.pkl')

    prectict = predict_values(model, df_horas, horas_pred)

    plt = plot_chart_pred(df_horas, prectict, horas_pred)

    st.plotly_chart(plt)


if __name__ == '__main__':
    main()
