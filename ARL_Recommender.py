# !pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
### ensures that the output is on a single line.
pd.set_option('display.expand_frame_repr', False)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

####
# Data Preprocessing
####

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = retail_data_prep(df)

## Invoice-Product Matrix

df_grm = df[df["Country"] == "Germany"]

grm_inv_pro_df = df_grm.pivot_table(index="Invoice", columns="StockCode", values="Quantity", aggfunc="sum").\
    fillna(0).\
    applymap(lambda x : 1 if x > 0 else 0)
grm_inv_pro_df.head()

df[df["StockCode"]== "POST"]
grm_inv_pro_df.drop("POST", axis=1, inplace=True) # Since "POST" is Post/Shipping fee, I will not take it to the union rule.

####
# Generate association rules through Germany customers.
####

grm_sup_val = apriori(grm_inv_pro_df, min_support=0.01, use_colnames=True)
grm_sup_val.sort_values("support", ascending=False).head(10)

grm_rules = association_rules(grm_sup_val, metric="support", min_threshold=0.01)


######
# Names of the products whose ID is given
######
def check_id(dataframe, stock_code):
    product_name = []
    if type(stock_code) == list:  # If the entered stockCode is a list, it returns the product names of the stockcodes in the list.
        for i in stock_code:
            product_name.append(dataframe[dataframe["StockCode"] == i][["Description"]].values[0].tolist())
    else:
        product_name.append(dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist())
    print(*product_name)

check_id(df, 21987)
# PACK OF 6 SKULL PAPER CUPS
check_id(df, 23235)
# STORAGE TIN VINTAGE LEAF
check_id(df, 22747)
# POPPY'S PLAYHOUSE BATHROOM

####
# Product recommendation for users at the basket stage
####

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_grm_rules = grm_rules.sort_values("support", ascending=False)
    recommendation_list = []

    for i, product in sorted_grm_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_grm_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

arl_recommender(grm_rules, 21987, 3)
arl_recommender(grm_rules, 23235, 3)
arl_recommender(grm_rules, 22747, 3)

check_id(df, arl_recommender(grm_rules, 21987, 3))
check_id(df, arl_recommender(grm_rules, 23235, 3))
check_id(df, arl_recommender(grm_rules, 22747, 3))