def count_qs_type(data_list):
    type_count = {}
    for item in data_list:
        type_count[item["type"]] = type_count.get(item["type"], 0) + 1
    return type_count
