# ----- print function -----
def print_data_num(data, data_name):
    print(data_name+" number:"+str(data.shape[0]))


def print_ttv_data_num(train_data,test_data,validate_data):
    print_data_num(train_data, "train_data")
    print_data_num(test_data, "test_data")
    print_data_num(validate_data, "validate_data")


def print_data_unique_context(data, data_name, col_name):
    unique_col = data[col_name].unique()
    print(data_name+"'s "+col_name+" unique context is:\n"+str(unique_col))
    print("number of "+data_name+"'s "+col_name+" unique context:"+str(len(unique_col)))


def print_ttv_data_unique_context(train_data, test_data, validate_data):
    print_data_unique_context(train_data, "train_data", "tag")
    print_data_unique_context(test_data, "test_data", "tag")
    print_data_unique_context(validate_data, "validate_data", "tag")

    print_data_unique_context(train_data, "train_data", "prefix")
    print_data_unique_context(test_data, "test_data", "prefix")
    print_data_unique_context(validate_data, "validate_data", "prefix")

    print_data_unique_context(train_data, "train_data", "title")
    print_data_unique_context(test_data, "test_data", "title")
    print_data_unique_context(validate_data, "validate_data", "title")
