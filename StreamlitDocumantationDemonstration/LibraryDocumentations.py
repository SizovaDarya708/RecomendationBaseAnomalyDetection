import streamlit as st
import random
import pandas as pd


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Главное меню":
        st.sidebar.title('Параметры модели LifgthFM')
        st.sidebar.markdown('no_components (int, необязательный) — размерность скрытых вложений функции.')
        no_components = st.sidebar.slider('no_components', 0, 100)
        params['no_components'] = no_components        
        st.sidebar.markdown('k (int, необязательно) — для обучения k-OS будет выбран k-й положительный '+
                            'пример из n положительных примеров, выбранных для каждого пользователя')
        k = st.sidebar.slider('k', 1, 50)
        params['k'] = k
        st.sidebar.markdown('n (int, необязательно) — для обучения k-OS максимальное количество положительных результатов для каждого обновления.'+
                            'пример из n положительных примеров, выбранных для каждого пользователя')
        n = st.sidebar.slider('n', 1, 50)
        params['n'] = n
        st.sidebar.markdown("epsilon: float, default=0.1. Эпсилон для функции потерь.")
        epsilon = st.sidebar.slider('epsilon', 0.1, 1.0)
        params['epsilon'] = epsilon
                
        st.sidebar.title('Параметры модели SVM')
        st.sidebar.markdown("C: float, default=1.0. Параметр регуляризации.")
        С = st.sidebar.slider('С', 1.0, 2.0)
        params['С'] = С
        st.sidebar.markdown("degree: int, default=3. Степень полиномиальной функции ядра.")
        degree = st.sidebar.slider('degree', 1, 10)
        params['degree'] = degree
        st.sidebar.markdown("cache_size: float, default=200. Размер кеша ядра (в МБ).")
        cache = st.sidebar.slider('cache', 100, 300)
        params['cache'] = cache 
        
        st.sidebar.title('Параметры модели KNN')
        st.sidebar.markdown("n_neighbors: int, default=5. Количество соседей для использования.")
        n_neighbors = st.sidebar.slider('n_neighbors', 1, 20)
        params['n_neighbors'] = n_neighbors
        st.sidebar.markdown("leaf_size: int, default=30. Размер листьев, влияет на скорость построения запроса, на объем памяти, необходимый для хранения дерева.")
        leaf_size = st.sidebar.slider('leaf_size', 10, 100)
        params['leaf_size'] = leaf_size
        st.sidebar.markdown("p: int, default=2. Степенный параметр для метрики Минковского.")
        p = st.sidebar.slider('p', 2, 10)
        params['p'] = p
        return params
        
def main_page(params): 
    
    df= pd.read_csv('train_12.csv',index_col=0,low_memory=False)
    st.write(df.head())
    
    if st.checkbox("Показать код"):
        st.code("""from lightfm import LightFM
    
    model_without_features.fit(user_to_product_interaction_train,
          user_features=None, 
          item_features=None, 
          sample_weight=None, 
          epochs=10, 
          num_threads=4,
          verbose=1)
            """)    
        st.code("""from sklearn.neighbors import KNeighborsClassifier

                knn = KNeighborsClassifier(n_neighbors=5);
                knn.fit(X_train, y_train);
                """)

        st.code("""from sklearn.svm import SVC
                clf = SVC(C=c, degree=deg, gamma='auto', cache_size=cache);
                clf.fit(X_train, y_train)
                """);
    
    
st.sidebar.header('Документация библиотеки')
page = st.sidebar.selectbox("Выбор функции библиотеки", ["Главное меню", "get_uniq_target_list", "recommendation_for_user","get_item_list",
                                                         "get_user_list","get_feature_list", "id_mappings", "get_user_product_interaction", "get_uniq_target_list", "data_preprocessing_by_feature_list", "get_anomaly_detection_results", "get_product_feature_interaction","get_interaction_matrix", "recommendation_for_many", "recommendation_base_anomaly_detection.fit", "recommendation_anomaly_detection.train"
                                                         "interactions"])
    
if page == "Главное меню":
    st.header("Сравнительная таблица моделей поиска аномалий")
    params = add_parameter_ui(page)
    main_page(params)
elif page == "get_uniq_target_list":
    st.header("get_uniq_target_list(df, target_column);")
    st.write("Получение уникальных, отсортированных атрибутов конкретного столбца таблицы.")
elif page == "recommendation_for_user":
        st.title("recommendation_for_user(user_id,n)")
        st.header("Получение рекомендаций для конкретного пользователя с помощью уже обученной модели рекомендательной системы")
        st.markdown("Входные параметры:")
        st.markdown("user_id(int, обязательный) - Идентификатор пользователя, которому необходимы рекомендации")
        st.markdown("n(int, обязательный) - Количество рекомендация пользователю")
        st.header("Пример использования")
        st.code("""recom.recommendation_for_user(669311, 2)""");
        st.write("Вывод данных")
        st.write("Пользователь 669311")        
        st.write("Совершенные действия:")          
        st.write("dc1-mail-03.ptsecurity.ru")           
        st.write("dc2-mail-224.ptsecurity.ru")           
        st.write("112-7.ptstorage.ru")           
        st.write("8098-490-76.ptsecurity.ru")        
        st.write("Рекомендации:")          
        st.write("dc1-mail-239.ptsecurity.ru")           
        st.write("dc2-mail-01.ptsecurity.ru")  
        
        
        st.header("Исходный код")
        st.code("""def recommendation_for_user(self, user, n):
        
        # получение индексов пользователя
        
        userindex = self.user2index_map.get(user, None)
        
        if userindex == None:
            return None
        
        users = userindex
        known_positives = self.items[self.user_to_product_interaction_matrix.tocsr()[userindex].indices]
        
        scores = self.model.predict(user_ids = users, item_ids = np.arange(self.user_to_product_interaction_matrix.shape[1]))
        
        top_items = self.items[np.argsort(-scores)]
        
        # Вывод результатов
        print("Пользователь %s" % user)
        print("     Совершенные действия:")
        
        for x in known_positives[:10]:
            print("                  %s" % x)
            
            
        print("     Рекомендации:")        
        for x in top_items[:10]:
            print("                  %s" % x)""")
elif page == "recommendation_base_anomaly_detection.fit":
        st.title("recommendation_base_anomaly_detection.fit(df, df_recommendations)")
        st.header("Передача данных модели поиска аномалий данных о рекомендациях объектов(пользователей).")
        st.markdown("Входные параметры:")
        st.markdown("df (обязательный, pandas.DataFrame) - двумерный массив, представляющий из себя таблицу обрабатываемых данных;")
        st.markdown("df_recommendations (обязательный, pandas.DataFrame) - двумерный массив, представляющий из себя таблицу соотношений объект-рекомендация.")
        st.markdown("Выходные параметры:")
        st.markdown("Функция без возвращаемого значения.(void)")
        st.header("Пример использования")
        st.markdown("Шаг 1. Получение df_recommendations")
        st.code("""import preprocess from DataPreprocessing;
        
        df_recommendations = preprocess.get_user_product_interaction(df, df['uid'], df['iid']);""");
        
        st.write("Вывод df_recommendations")
        df = pd.DataFrame({
        'Идентификатор пользователя (uid)': [3456, 3456, 3980],
        'Идентификатор объекта рекомендаций (iid)': [95346, 45867, 96738]
        })
        st.write(df);
        st.write("Шаг 2. Передача модели поиска аномалий данных")        
        st.code("""import anomaly_detection from RecommendationBaseAnomalyDetectionModel;
        
        anomaly_detection.fit(df, df_recommendations);""");    

#посмотреть какие параметры можно изменять и настраивать в моделях
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        n_start = st.sidebar.slider('n_start', 10, 200)
        params['n_start'] = n_start
        n_stop = st.sidebar.slider('n_stop', 300, 500)
        params['n_stop'] = n_stop
        n_num = st.sidebar.slider('n_num', 50, 1000)
        params['n_num'] = n_num
    elif clf_name == "LightGBM":
        st.sidebar.markdown("num_leaves: int, default=31. Максимальное количество листьев на дереве для базового обучения.")
        num_leaves = st.sidebar.slider('num_leaves', 30, 100)
        params['num_leaves'] = num_leaves
        st.sidebar.markdown("n_estimators: int, default=100. Количество boosted trees для обучения.")
        n_estimators = st.sidebar.slider('n_estimators', 50, 200)
        params['n_estimators'] = n_estimators
        st.sidebar.markdown("min_child_samples: int, default=20. Минимальное количество данных, необходимых для потомка/листа.")
        min_child_samples = st.sidebar.slider('min_child_samples', 1, 10)
        params['min_child_samples'] = min_child_samples
    elif clf_name == "Stochastic Gradient Decent":
        st.sidebar.markdown("alpha: float, default=0.0001. Константа, умножающая член регуляризации.")
        al = st.sidebar.slider('alpha', 0.01, 0.1)
        params['al'] = al
        st.sidebar.markdown("epsilon: float, default=0.1. Эпсилон для функции потерь.")
        epsilon = st.sidebar.slider('epsilon', 0.1, 1.0)
        params['epsilon'] = epsilon
        st.sidebar.markdown("eta0: double, default=0.0. Начальная скорость обучения.")
        eta = st.sidebar.slider('eta', 0.0, 1.0)
        params['eta'] = eta
        st.sidebar.markdown("n_iter_no_change: int, default=5. Количество итераций без улучшений, чтобы дождаться досрочной остановки.");
        n_iter = st.sidebar.slider('n_iter', 1, 10)
        params['n_iter'] = n_iter         
    return params




    