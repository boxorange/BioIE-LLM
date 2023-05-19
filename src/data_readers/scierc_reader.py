"""

- SciERC data contains samples with no relations.

"""
import os
import json


class SciercReader:
    
    def __init__(self, path, task=None):
        path = os.path.expanduser(path)
        
        orig_data_dir = os.path.join(path, "SciERC/spert")
        converted_data_dir = os.path.join(path, "SciERC/converted")
        type_file = os.path.join(path, "SciERC/spert/scierc_types.json")
        
        if not os.path.exists(converted_data_dir):
           os.makedirs(converted_data_dir)
           
        if len(os.listdir(converted_data_dir)) == 0:
            self._convert_data(orig_data_dir, converted_data_dir)
            
        self.train_data = json.load(open(os.path.join(converted_data_dir, "train.json")))
        self.test_data = json.load(open(os.path.join(converted_data_dir, "test.json")))
                
        type_data = json.load(open(type_file))

        self.ent_types = [x for x in type_data["entities"]]
        self.rel_types = [x for x in type_data["relations"]]


    def _convert_data(self, orig_data_dir, converted_data_dir):
        
        for file in os.listdir(orig_data_dir):
            if file not in ['scierc_train_dev.json', 'scierc_test.json']:
                continue
            
            output_data = []
            
            data = json.load(open(os.path.join(orig_data_dir, file)))
            
            for item in data:
                tokens = item['tokens']
                entities = item['entities']
                relations = item['relations']
                orig_id = item['orig_id']
                
                text = ' '.join(tokens)
                entities = [(' '.join(tokens[e['start']:e['end']]), e['type']) for e in entities]
                relations = [(entities[r['head']][0], entities[r['tail']][0], r['type']) for r in relations]

                converted = {"id": orig_id,
                             "text": text,
                             "entities": entities,
                             "relations": relations,
                             "directed": True} # relation directionality. a.k.a symmetric or asymmetric relation.
                                               # For testing phase, undirected samples are replicated, and the replicated samples are tagged as reverse. 
                                               # So, if it's set to true, the model uses the second entity + the first entity instead of 
                                               # the first entity + the second entity to classify both relation representation cases (A + B, B + A). 
            
                output_data.append(converted)
            
            outfile = ''
            if file == 'scierc_train_dev.json':
                outfile = 'train.json'
            elif file == 'scierc_test.json':
                outfile = 'test.json'
                
            outfile = os.path.join(converted_data_dir, outfile)

            with open(outfile, "w") as fout:
                #f.write(output_data)
                json.dump(output_data, fout)

    '''
    @classmethod
    def get_data(self):
        if len(os.listdir(self.converted_data_dir)) == 0:
            self.convert_data(self)
            
        train_data = json.load(open(os.path.join(self.converted_data_dir, "train.json")))
        test_data = json.load(open(os.path.join(self.converted_data_dir, "test.json")))
        
        return train_data, test_data
    
    
    @classmethod
    def get_type(self):	
        type_data = json.load(open(self.type_file))
        
        #print(type_data["entities"])
        #for x in type_data["entities"].values():
        #	print(x)
        #	input('enter..')

        ent_types = [x for x in type_data["entities"]]
        #ent_types = [x["verbose"] for x in type_data["entities"].values()] # read "verbose" since it's more readable and understandable.
        rel_types = [x for x in type_data["relations"]]
        
        return ent_types, rel_types
    '''