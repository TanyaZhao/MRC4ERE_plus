question_templates = {
    "located_in": ["Find locations which {} is located in .",
                   "Which location does {} belong to ?",
                   "Where does {} located in ?"],
    "work_for": ["Find organizations which {} worked for .",
                 "Which organization is {} working for ?",
                 "Where does {} work for ?"],
    "live_in": ["Find locations which {} is lived in .",
                "Where does {} live ?",
                "Where is {}'s home ?"],
    "kill": ["Find people who is killed by {} .",
"Which person is killed by {} ?",
             "Who is killed by {} ?"],
    "orgbased_in": ["Find locations where {} is based in .",
                    "What is the location of {} ?",
                    "Where is {} ?"],
}

entity_relation_map = {
    "loc": ["located_in"],
    "peop": ["work_for", "live_in", "kill"],
    "org": ["orgbased_in"],
    "other": []
}