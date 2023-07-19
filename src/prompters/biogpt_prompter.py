
class BioGPTPrompter:
    def get_string_prompt(self):
        """
        - Question, Answer produce different inferences from Q, A. 02/09/2023
        - There shouldn't be any space at the end of query text. E.g., "Answer: " -> the model generates nothing. 02/14/2023
        
        """
        #entity_q = lambda entity: "\n\nQuestion: Which proteins are bound to {x}?\n\n".format(x=entity)
        #entity_q = lambda entity: "\n\nQuestion: What proteins are bound to {x}?\n\n".format(x=entity)
        #entity_q = lambda entity: "\n\nQ: What proteins are bound to {x}?\n\n".format(x=entity)
        #entity_q = lambda entity: "\n\nQ: What proteins does {x} bind to?\n\n".format(x=entity)
        #entity_q = lambda entity: "\n\nQ: To what proteins does {x} bind?\n\n".format(x=entity)
        entity_q = lambda entity: "Question: Which proteins are related to {x}?\n\nAnswer:".format(x=entity)
        #entity_q = lambda entity: "Question: Which proteins interact with {x}?\n\nAnswer:".format(x=entity)
        #entity_a = lambda entities: "A: {x}\n\n".format(x=entities)
        #entity_q = lambda entity: "The following proteins are related to \"{x}\": ".format(x=entity)
        #entity_q = lambda entity: "The following proteins are bound to the protein \"{x}\": ".format(x=entity)
        #entity_q = lambda entity: "The following proteins interact with the protein \"{x}\": ".format(x=entity)
        entity_a = lambda entities: " {x}\n\n".format(x=entities)
        
        #relation_q = lambda e1, e2: "Question: Do the two proteins \"{x}\" and \"{y}\" bind each other?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do the two proteins {x} and {y} bind each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do the two proteins {x} and {y} bind to each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do {x} and {y} bind each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Does {x} bind to {y}? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do {x} and {y} bind to each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Are {x} and {y} related to each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Are {x} and {y} related to each other?\n\nAnswer:".format(x=e1, y=e2)
        relation_q = lambda e1, e2: "Question: Do {x} and {y} interact with each other?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Are {x} and {y} related to each other? yes or no\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: {x} and {y} are related to each other. Is this statement True or False?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: {x} and {y} are related to each other.\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "\n\nQuestion: Given the options: \"Related\", \"Unrelated\", which one is the relation type between {x} and {y}?\n\nAnswer:".format(x=e1, y=e2)
        relation_a = lambda answer: " {x}\n\n".format(x=answer)
        
        return {'entity_q': entity_q,
                'entity_a': entity_a,
                'relation_q': relation_q,
                'relation_a': relation_a}
    
    
    def get_kegg_prompt(self):
        entity_q = lambda pathway: "Question: Which genes are involved in \"{x}\"?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes are involved in {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes are related to {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which proteins are related to {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes or proteins are related to {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes/proteins are related to {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "The following genes are involved in \"{x}\" pathway: ".format(x=pathway)
        #entity_q = lambda pathway: "The genes involved in \"{x}\" pathway are".format(x=pathway) # precision: 0
        entity_a = lambda entities: " {x}\n\n".format(x=entities)
        
        #relation_q = lambda e1, e2: "Question: Are {x} and {y} related to each other?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Are \"{x}\" and \"{y}\" related to each other?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Is {x} related to {y}?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Is {x} related to the pathway {y}?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Is {x} involved in {y}?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Is \"{x}\" involved in \"{y}\"?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Is {x} involved in the human pathway {y}?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Is {x} involved in the KEGG pathway {y}?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Does \"{y}\" have \"{x}\"?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "The two genes \"{x}\" and \"{x}\" are ".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Is this statement \"The two genes \"{x}\" and \"{x}\" are related\" true? ".format(x=e1, y=e2)
        #relation_a = lambda answer: "{x}\n\n".format(x=answer)
        #relation_q = lambda e1, e2: "QUESTION: Given the options: \"related\", \"unrelated\", which one is the relation type between {x} and {y}? ".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "QUESTION: which one is the relation type between {x} and {y}? choose the correct answer among the following options: \"related\", \"unrelated\" ".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "QUESTION: Are \"{x}\" and \"{y}\" related to each other? Choose the correct answer among the following options: \"yes\", \"no\"\n\nANSWER: ".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Q: Are \"{x}\" and \"{y}\" related to each other?\n".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Q: Are the two genes \"{x}\" and \"{y}\" related to each other? Choose the correct answer among the following options: \"yes\", \"no\"\n".format(x=e1, y=e2)
        relation_q = lambda e1, e2: "Question: Is the gene \"{x}\" involved in the pathway \"{y}\"?\n\nAnswer:".format(x=e1, y=e2)
        relation_a = lambda answer: " {x}\n\n".format(x=answer) # W/o separator token '</s>' BioGPT-Large prepends "Answer: yes" e.g., "Answer: yes Question: No" 05/27/2023
        
        return {'entity_q': entity_q,
                'entity_a': entity_a,
                'relation_q': relation_q,
                'relation_a': relation_a}


    def get_indra_prompt(self):
        #relation_type_q = lambda e1, e2, choices: "\n\nQuestion: Which of the following is the relation type between {x} and {y} in the text above? {z}\n\nAnswer:".format(x=e1, y=e2, z=choices)
        #relation_type_q = lambda e1, e2, choices: "\n\nQuestion: Which of the following is the relation type between \"{x}\" and \"{y}\" in the text above? {z}\n\nAnswer:".format(x=e1, y=e2, z=choices)
        #relation_type_q = lambda e1, e2, choices: "\n\nQuestion: Which of the following is the relation between \"{x}\" and \"{y}\" in the text above? {z}\n\nAnswer:".format(x=e1, y=e2, z=choices)
        #relation_type_a = lambda num, relation_type: " ({x}) {y}\n\n".format(x=num, y=relation_type)
        
        #relation_type_q = lambda e1, e2, choices: "\n\nQuestion: Given the options: {z}, which one is the relation type between \"{x}\" and \"{y}\" in the text above?\n\nAnswer:".format(x=e1, y=e2, z=choices)
        relation_type_q = lambda e1, e2, choices: "\n\nQuestion: Given the options: {z}, which one is the relation type between {x} and {y} in the text above?\n\nAnswer:".format(x=e1, y=e2, z=choices)
        #relation_type_q = lambda e1, e2, choices: "Question: Given the options: {z}, which one is the relation type between \"{x}\" and \"{y}\" in the text above?\n\nAnswer:".format(x=e1, y=e2, z=choices)
        
        # testing for question, text (context), answer format used in PubMedQA - 05/31/2023
        # but, the results were worse than the original format.
        #relation_type_q = lambda e1, e2, text, choices: "Question: Given the options: {z}, which one is the relation type between \"{x}\" and \"{y}\" in this text \"{t}\"?\n\nAnswer:".format(x=e1, y=e2, t=text, z=choices)
        relation_type_a = lambda relation_type: " {x}\n\n".format(x=relation_type)
        
        return {'relation_type_q': relation_type_q,
                'relation_type_a': relation_type_a}
