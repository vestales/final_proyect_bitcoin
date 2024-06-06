import streamlit as st
from Funciones import plot_chart, descargar_datos, cargar_limpiar_datos, crear_tabla_por_dia, crear_tabla_por_horas

def main():
    st.title('Price of bitcoin')

    st.write("### Text here")

    url = descargar_datos()

    df = cargar_limpiar_datos(url)

    df_horas = crear_tabla_por_horas(df)

    df_dias = crear_tabla_por_dia(df)

    plt = plot_chart(df_dias,'BTC/USDT Minute-by-Minute Prices').show()
    
    st.pyplot(plt)


    


if __name__ == '__main__':
    main()
