from Funciones import descargar_datos, cargar_limpiar_datos, crear_tabla_por_horas, crear_tabla_por_dia, crear_modelo

url = descargar_datos()

df = cargar_limpiar_datos(url)

df_horas = crear_tabla_por_horas(df)

df_dias = crear_tabla_por_dia(df)

predictions = crear_modelo(df_horas)
