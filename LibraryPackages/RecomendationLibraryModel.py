class RecommendationBaseAnomalyDetectionModel:
    def get_interaction_matrix(df, df_column_as_row, df_column_as_col, df_column_as_value, row_indexing_map,
                          col_indexing_map):
    
    row = df[df_column_as_row].apply(lambda x: row_indexing_map[x]).values
    col = df[df_column_as_col].apply(lambda x: col_indexing_map[x]).values
    value = df[df_column_as_value].values
    
    return coo_matrix((value, (row, col)), shape = (len(row_indexing_map), len(col_indexing_map)))

    def __init__(self, model, items = items, user_to_product_interaction_matrix = user_to_product_interaction_train, 
                user2index_map = user_to_index_mapping):
        
        self.user_to_product_interaction_matrix = user_to_product_interaction_matrix
        self.model = model
        self.items = items
        self.user2index_map = user2index_map
    
    def recommendation_for_user(self, user):
        
        # получение индексов пользователя
        
        userindex = self.user2index_map.get(user, None)
        
        if userindex == None:
            return None
        
        users = userindex
        
        # хосты, которые уже были посещены
        
        known_positives = self.items[self.user_to_product_interaction_matrix.tocsr()[userindex].indices]
        
        scores = self.model.predict(user_ids = users, item_ids = np.arange(self.user_to_product_interaction_matrix.shape[1]))
        
        # топ переходов
        
        top_items = self.items[np.argsort(-scores)]
        
        # Вывод результатов
        print("Пользователь %s" % user)
        print("     Уже известные посещаемые хосты:")
        
        for x in known_positives[:10]:
            print("                  %s" % x)
            
            
        print("     Рекомендации:")
        
        for x in top_items[:10]:
            print("                  %s" % x)

    def recommendation_for_many(self, users):

      predictions = []

      for i in users:
        user = self.user2index_map.get(i)
        scores = self.model.predict(user_ids = int(user), item_ids = np.arange(self.user_to_product_interaction_matrix.shape[1]))
        top_items = self.items[np.argsort(-scores)]
        predictions.append(top_items[:10].tolist())
      
      return predictions
    
    def get_product_feature_interaction(product_df, aisle_df, department_df, aisle_weight = 1, department_weight = 1):
        item_feature_df = product_df.merge(aisle_df).merge(department_df)[["product_name", "aisle", "department"]]

        # Индексирование данных
        item_feature_df["product_name"] = item_feature_df["product_name"]
        item_feature_df["aisle"] = item_feature_df["aisle"]
        item_feature_df["department"] = item_feature_df["department"]    

        product_aisle_df = item_feature_df[["product_name", "aisle"]].rename(columns = {"aisle" : "feature"})
        product_aisle_df["feature_count"] = aisle_weight 
        product_department_df = item_feature_df[["product_name", "department"]].rename(columns = {"department" : "feature"})
        product_department_df["feature_count"] = department_weight     

        product_feature_df = pd.concat([product_aisle_df, product_department_df], ignore_index=True)

        del item_feature_df
        del product_aisle_df
        del product_department_df    

        # группировка для суммирования по feature_count
        product_feature_df = product_feature_df.groupby(["product_name", "feature"], as_index = False)["feature_count"].sum()    

        return product_feature_df

    def get_user_product_interaction(user_to_product):

        user_to_product["product_count"] = 1
        user_to_product_rating_train = user_to_product.groupby(["uid", "iid"], as_index = False)["product_count"].sum()

        return user_to_product_rating_train
    
    def interactions(data, row, col, value, row_map, col_map):
    
        #converting the row with its given mappings
        row = data[row].apply(lambda x: row_map[x]).values
        #converting the col with its given mappings
        col = data[col].apply(lambda x: col_map[x]).values
        value = data[value].values
        #returning the interaction matrix
    return coo_matrix((value, (row, col)), shape = (len(row_map), len(col_map)))