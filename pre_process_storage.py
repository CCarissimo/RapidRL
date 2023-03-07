import pickle

storage = {}

for size_stories in [0, 1, 2, 5, 7, 15]:

    for reset_buffer_type in ["death", "random", "both"]:

        for reset_q_table in [True, False]:

            for reset_visits in [True, False]:

                with open(f"storage/tables_storage_stories_size{size_stories}_resetQ{reset_q_table}_resetV{reset_visits}_"
                          f"resetB{reset_buffer_type}", "rb") as file:
                    data = pickle.load(file)
                    storage[(size_stories, reset_q_table, reset_visits, reset_buffer_type)] = data["metrics"]
                    print((size_stories, reset_q_table, reset_visits, reset_buffer_type))

print(storage)

with open("storage/merged_storage", "wb") as file:
    pickle.dump(storage, file)
