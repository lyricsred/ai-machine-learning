import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

st.title('Универсальное EDA и ML-приложение')

st.header('Шаг 1: Загрузка данных (CSV)')
uploaded_file = st.file_uploader('Выберите CSV-файл с признаками и целевой переменной', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('Первые 5 строк данных:')
    st.write(df.head())
    st.write(f'Ваша таблица: {df.shape[0]} объектов, {df.shape[1]} признаков')

    st.header('Шаг 2: Выбор целевой переменной')
    target_column = st.selectbox('Выберите колонку-целевую переменную для обучения', df.columns)
    feature_columns = [col for col in df.columns if col != target_column]

    unique_cols = [c for c in feature_columns if df[c].nunique() == len(df)]
    if unique_cols:
        st.info(f'Удалены явно неинформативные колонки (уникальные): {unique_cols}')
        feature_columns = [c for c in feature_columns if c not in unique_cols]

    temp_df = df.copy()
    for col in feature_columns:
        if temp_df[col].dtype in ['int64', 'float64']:
            temp_df[col] = temp_df[col].fillna(temp_df[col].mean())
        else:
            temp_df[col] = temp_df[col].fillna(temp_df[col].mode()[0])

    X = pd.get_dummies(temp_df[feature_columns], drop_first=True)
    feature_names = list(X.columns)

    if temp_df[target_column].dtype == 'object':
        y = temp_df[target_column].astype('category').cat.codes
    else:
        y = temp_df[target_column]

    task_type = 'regression'
    if y.nunique() <= 10 and set(np.unique(y)) <= set(range(10)):
        task_type = st.radio('Тип задачи', ['classification', 'regression'], index=0)
    else:
        task_type = st.radio('Тип задачи', ['regression', 'classification'], index=0)
    st.write(f'Обнаружено признаков: {X.shape[1]}. Тип задачи: {task_type}')

    st.header('Шаг 3: Графики EDA')

    st.subheader('Гистограммы')
    selected_col = st.selectbox('Выберите признак для гистограммы', feature_names)
    fig, ax = plt.subplots()
    sns.histplot(X[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader('Корреляционная матрица (heatmap)')
    if X.shape[1] <= 25:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)
    else:
        st.write('Слишком много признаков для корреляционного heatmap (>25).')

    st.subheader('Диаграмма рассеяния')
    if len(feature_names) >= 2:
        scatter_x = st.selectbox('X признак', feature_names, index=0, key='xscat')
        scatter_y = st.selectbox('Y признак', feature_names, index=1, key='yscat')
        fig3, ax3 = plt.subplots()
        sc = ax3.scatter(X[scatter_x], X[scatter_y], c=y, cmap='coolwarm')
        plt.xlabel(scatter_x)
        plt.ylabel(scatter_y)
        plt.colorbar(sc)
        st.pyplot(fig3)
    else:
        st.write('Недостаточно числовых признаков для scatterplot.')

    st.header('Шаг 4: Обучение и тестирование ML-модели')
    test_size = st.slider('Размер тестовой выборки', 0.1, 0.5, 0.2)
    random_state = st.number_input('random_state', value=42)

    if task_type == 'classification':
        model = LogisticRegression(max_iter=500)
    else:
        model = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    model.fit(X_train, y_train)

    if task_type == 'classification':
        score = model.score(X_test, y_test)
        st.write(f'Точность (accuracy) на тестовой выборке: {score:.3f}')
    else:
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        st.write(f'R² на тестовой выборке: {score:.3f}')

    st.header('Шаг 5: Предсказание по новым данным')
    st.write('Введите значения признаков (любое количество отсутствующих будет заменено средними/модой):')
    user_input = {}
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            default_val = float(df[col].mean())
            user_input[col] = st.number_input(f'{col}', value=default_val)
        else:
            example = str(df[col].mode()[0])
            user_input[col] = st.text_input(f'{col}', value=example)

    if st.button('Сделать предсказание'):
        input_df = pd.DataFrame([user_input])
        for col in feature_columns:
            if input_df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    input_df[col] = input_df[col].fillna(df[col].mean())
                else:
                    input_df[col] = input_df[col].fillna(df[col].mode()[0])
        input_X = pd.get_dummies(input_df, drop_first=True)
        for f in feature_names:
            if f not in input_X.columns:
                input_X[f] = 0
        input_X = input_X[feature_names]

        pred = model.predict(input_X)[0]
        st.success(f'Предсказание модели: {pred}')

    st.header('Шаг 6: Визуализация важности/весов признаков')
    if hasattr(model, 'coef_'):
        weights = model.coef_
        if weights.ndim > 1:
            weights = weights[0]
        fig4, ax4 = plt.subplots()
        inds = np.argsort(np.abs(weights))[::-1]
        sns.barplot(y=np.array(feature_names)[inds][:20], x=weights[inds][:20], orient='h', ax=ax4)
        plt.title('Коэффициенты (важность признаков) модели')
        st.pyplot(fig4)
    else:
        st.write('Модель не поддерживает коэффициенты признаков.')

else:
    st.info('Пожалуйста, загрузите CSV-файл.')
