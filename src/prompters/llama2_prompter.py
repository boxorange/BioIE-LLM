
class Llama2Prompter:
    def get_scierc_prompt(self):
        entity_q = "\n\nQ: What scientific entities are mentioned in this text?\n\n"
        entity_a = lambda entities: "A: {x}\n\n".format(x=entities)
        
        #entity_type_q = lambda entity, choices: "Q: Which one of the following is the entity type of \"{x}\"? {y} \n\n".format(x=entity, y=choices)
        entity_type_q = lambda entity, choices: "\n\nQ: Which of the following is the entity type of \"{x}\"? {y}\n\n".format(x=entity, y=choices)
        entity_type_a = lambda num, entity_type: "A: ({x}) {y}\n\n".format(x=num, y=entity_type)
                    
        #relation_type_q = lambda e1, e2, choices: "Q: What is the relation between between \"{x}\" and \"{y}\"? {z}\n\n".format(x=e1, y=e2, z=choices)
        relation_type_q = lambda e1, e2, choices: "\n\nQ: Which of the following is the relation type between \"{x}\" and \"{y}\"? {z}\n\n".format(x=e1, y=e2, z=choices)
        relation_type_a = lambda num, relation_type: "A: ({x}) {y}\n\n".format(x=num, y=relation_type)
        
        return {'entity_q': entity_q,
                'entity_a': entity_a,
                'entity_type_q': entity_type_q,
                'entity_type_a': entity_type_a,
                'relation_type_q': relation_type_q,
                'relation_type_a': relation_type_a}

    
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
        #entity_q = lambda entity: "\n\nQuestion: Which proteins are related to {x}?\n\nAnswer:".format(x=entity)
        entity_q = lambda entity: "QUESTION: Which proteins interact with {x}?\nANSWER:".format(x=entity)
        entity_a = lambda entities: " {x}\n".format(x=entities)
        #entity_q = lambda entity: "[INST] Which proteins interact with {x}?[/INST]\n".format(x=entity)
        #entity_a = lambda entities: "{x}\n".format(x=entities)
        #entity_q = lambda entity: "The following proteins are related to \"{x}\": ".format(x=entity)
        #entity_q = lambda entity: "The following proteins are bound to the protein \"{x}\": ".format(x=entity)
        #entity_q = lambda entity: "The following proteins interact with the protein \"{x}\":".format(x=entity)
        #entity_a = lambda entities: " {x}\n\n".format(x=entities)
        
        
        
        #entity_q = lambda entity: "### User:\nWhich proteins interact with {x}?\n\n### Assistant:\n".format(x=entity)
        #entity_a = lambda entities: "{x}\n\n".format(x=entities)
        
        
        
        
        
        #relation_q = lambda e1, e2: "Question: Do the two proteins \"{x}\" and \"{y}\" bind each other?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do the two proteins {x} and {y} bind each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do the two proteins {x} and {y} bind to each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do {x} and {y} bind each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Does {x} bind to {y}? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Do {x} and {y} bind to each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Are {x} and {y} related to each other? True or False\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Are {x} and {y} related to each other?\n\nAnswer:".format(x=e1, y=e2)
        relation_q = lambda e1, e2: "QUESTION: Do {x} and {y} interact with each other?\nANSWER:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "QUESTION: Does {x} interact with {y}?\nANSWER:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: Are {x} and {y} related to each other? yes or no\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: {x} and {y} are related to each other. Is this statement True or False?\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "Question: {x} and {y} are related to each other.\n\nAnswer:".format(x=e1, y=e2)
        #relation_q = lambda e1, e2: "\n\nQuestion: Given the options: \"Related\", \"Unrelated\", which one is the relation type between {x} and {y}?\n\nAnswer:".format(x=e1, y=e2)
        relation_a = lambda answer: " {x}\n".format(x=answer)
        
        return {'entity_q': entity_q,
                'entity_a': entity_a,
                'relation_q': relation_q,
                'relation_a': relation_a}
    
    
    def get_kegg_prompt(self):
        #entity_q = lambda pathway: "Question: Which genes are involved in \"{x}\"?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes are involved in {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes are related to {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which proteins are related to {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes or proteins are related to {x}?\n\nAnswer:".format(x=pathway)
        #entity_q = lambda pathway: "Question: Which genes/proteins are related to {x}?\n\nAnswer:".format(x=pathway)
        entity_q = lambda pathway: "The following genes are involved in \"{x}\" pathway:".format(x=pathway)
        #entity_q = lambda pathway: "The genes involved in \"{x}\" pathway are".format(x=pathway) # precision: 0
        entity_a = lambda entities: " {x}\n\n".format(x=entities)
        
        
        #entity_q = lambda pathway: "QUESTION: Which genes interact with \"{x}\"?\nANSWER:".format(x=pathway)
        #entity_q = lambda pathway: "QUESTION: Which genes are involved in \"{x}\"?\nANSWER:".format(x=pathway)
        entity_q = lambda pathway: "QUESTION: Which genes are associated with \"{x}\"?\nANSWER:".format(x=pathway)
        #entity_q = lambda pathway: "QUESTION: Which genes are associated with \"{x}\" pathway?\nANSWER:".format(x=pathway)
        #entity_q = lambda pathway: "QUESTION: Which genes and proteins are associated with \"{x}\"?\nANSWER:".format(x=pathway)
        #entity_q = lambda pathway: "QUESTION: Which genes and proteins are associated with \"{x}\" pathway?\nANSWER:".format(x=pathway)
        #entity_q = lambda pathway: "QUESTION: Which molecular objects are associated with \"{x}\"?\nANSWER:".format(x=pathway)
        entity_a = lambda entities: " {x}\n".format(x=entities)
        
        # entity_q = lambda pathway: "QUESTION: Which genes are related to \"{x}\"?\nANSWER:".format(x=pathway)
        # entity_a = lambda entities: " {x}\n".format(x=entities)
        
        
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
        relation_a = lambda answer: " {x}\n\n".format(x=answer)
        
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
        relation_type_a = lambda relation_type: " {x}\n\n".format(x=relation_type)
        
        return {'relation_type_q': relation_type_q,
                'relation_type_a': relation_type_a}
    
    
    def get_kbase_prompt(self):
        # entity_type_q = lambda e, choices: "\n\nQuestion: Given the options: {z}, which one is the entity type of {x} in the text above?\n\nAnswer:".format(x=e, z=choices)
        # entity_type_a = lambda relation_type: " {x}\n\n".format(x=relation_type)
        
        entity_type_q = lambda e, t, choices: '''Here is the definition of the entities.
        
A plasmid, in the context of genetic tools, is a small, circular DNA molecule separate from the chromosomal DNA. It is commonly used in genetic engineering and biotechnology as a vector to carry and replicate foreign genes. Plasmids are versatile tools for introducing specific genes into organisms for various purposes, such as gene expression studies, protein production, or modifying the traits of an organism.

The term "organism host" in genetic tools typically refers to the living organism in which a particular genetic manipulation or experiment is conducted. It is the host organism that carries and expresses the manipulated genetic material. In the context of genetic tools, the organism host is crucial for studying gene function, protein expression, or other genetic processes. The choice of organism host depends on the specific goals of the experiment, and commonly used hosts include bacteria, yeast, plants, and animals.

In genetic tools, a promoter is a DNA sequence that initiates the transcription of a particular gene. It serves as a binding site for RNA polymerase and other transcription factors, facilitating the start of the transcription process. The promoter region plays a crucial role in regulating gene expression by controlling when and how often a gene is transcribed into messenger RNA (mRNA), which is a key step in protein synthesis.

Genome engineering in genetic tools refers to the precise manipulation or modification of an organism's DNA to achieve desired changes in its genetic makeup. This process involves techniques such as CRISPR-Cas9, TALENs, or zinc finger nucleases to add, delete, or alter specific genetic elements within an organism's genome. The goal is to study gene function, model diseases, or engineer organisms with improved traits.

In genetic tools, the cloning method involves the process of creating identical copies of a gene, DNA fragment, or entire organism. This method allows for the replication and manipulation of genetic material for various purposes, including gene expression studies, production of specific proteins, and genetic engineering.

In the context of genetic tools, a "reporter" refers to a gene or a sequence of DNA that produces a detectable signal, often a measurable and visible output such as fluorescence or luminescence. Reporters are commonly used in molecular biology and genetics to monitor the activity of other genes, study gene expression patterns, or assess the effectiveness of genetic modifications. The signal produced by the reporter gene provides insights into the activity or behavior of the genetic elements under investigation.

In the context of genetic tools, a "regulator" typically refers to a molecule or a genetic element that controls the expression of genes. It can either enhance (activator) or inhibit (repressor) the transcription of specific genes, thereby influencing the synthesis of proteins and the regulation of cellular processes. Regulatory elements play a crucial role in gene expression and the overall functioning of genetic tools.

An antibiotic marker in genetic tools is a selectable gene introduced into an organism's DNA alongside a gene of interest. This marker, often conferring resistance to a specific antibiotic, allows researchers to identify and select for organisms that have successfully incorporated the desired genetic material, aiding in the screening and isolation of transformed cells.

A genetic screen in genetic tools refers to a systematic method used to identify and analyze genes associated with specific traits or functions within an organism's genome. It involves the deliberate alteration or manipulation of genes, followed by the observation of resulting phenotypic changes. Genetic screens are instrumental in understanding gene function, uncovering genetic interactions, and identifying genes involved in various biological processes.

In genetic tools, "RBS" stands for "Ribosome Binding Site." It is a short sequence of nucleotides in mRNA that provides a binding site for a ribosome during the initiation of translation, facilitating the assembly of the ribosome on the mRNA and the subsequent synthesis of a protein. The RBS is a crucial element in genetic engineering and synthetic biology, as it influences the efficiency of translation and protein expression in a given genetic construct.

Counter selection, in the context of genetic tools, refers to a technique used to selectively eliminate or suppress certain genotypes or phenotypes while favoring the survival or proliferation of others. This method is often employed in genetic engineering and molecular biology to achieve specific modifications or isolate desired traits within a population of organisms. Counter selection typically involves the use of negative selective pressure, such as the introduction of a lethal gene or a condition that is detrimental to the organisms carrying certain characteristics, aiming to encourage the survival of those with the desired traits.

In the context of genetic tools, a "terminator" refers to a DNA sequence that signals the end of a gene or a transcription unit. It serves as a signal to the cellular machinery that transcribes DNA (RNA polymerase) to stop transcription at that point. Terminators are crucial elements in genetic engineering and synthetic biology, as they help define the boundaries of genetic constructs and ensure that the genetic information is accurately transcribed and translated into proteins.

DNA transfer in genetic tools refers to the process of introducing foreign DNA into a host organism's cells. This transfer is a crucial step in genetic engineering and biotechnology, allowing researchers to manipulate the genetic material of organisms for various purposes, such as gene expression studies, protein production, or the creation of genetically modified organisms (GMOs). The methods of DNA transfer include techniques like transformation, transfection, electroporation, and viral vectors, depending on the type of host organism and the goals of the genetic modification.

In the context of genetic tools, an "operator" typically refers to a specific DNA sequence to which a regulatory protein binds, influencing the expression of adjacent genes. It is a short DNA segment that serves as a binding site for regulatory proteins, such as transcription factors. The interaction between the operator and regulatory proteins plays a crucial role in controlling gene expression by either promoting or inhibiting transcription.
\n\nQuestion: Given the options: {z}, which one is the entity type of {x} in this text? Text: {y}\n\nAnswer:'''.format(x=e, y=t, z=choices)
        
        entity_type_q = lambda e, t, choices: "Question: Given the options: {z}, which one is the entity type of {x} in this text? Text: {y}\n\nAnswer:".format(x=e, y=t, z=choices)
        
        entity_type_a = lambda relation_type: " {x}\n\n".format(x=relation_type)
        
        
        #entyty_and_entity_type_q = lambda choices: "\n\nQuestion: What are the biological entities and their types among the options: {z}?\n\nAnswer:".format(z=choices)
        entyty_and_entity_type_q = lambda choices: "\n\nQuestion: What are the biological entities and genetic tools in the text above?\n\nAnswer:".format(z=choices)
        entyty_and_entity_type_a = lambda answer: " {x}\n".format(x=answer)
        
        return {'entity_type_q': entity_type_q,
                'entity_type_a': entity_type_a,
                'entyty_and_entity_type_q': entyty_and_entity_type_q,
                'entyty_and_entity_type_a': entyty_and_entity_type_a}
    
    
    def get_lll_prompt(self):
        """
        
        """

        # relation_type_q = lambda e1, e2, t, c: "Question: Given the options: {c}, which one is the relation type between {x} and {y} in this text? Text: {z}\n\nAnswer:".format(x=e1, y=e2, z=t, c=c)
        relation_type_q = lambda e1, e2, t, c: "{z}\n\nQuestion: Given the options: {c}, which one is the relation type between {x} and {y} in the text above?\n\nAnswer:".format(x=e1, y=e2, z=t, c=c)
        
        # relation_type_q = lambda e1, e2, t, c: "Question: we find {x} and {y}. Among the choices {c}, what term best describes the type of relationship between them in this text? Text:{z}\n\nAnswer:".format(x=e1, y=e2, z=t, c=c)
        
        
        # relation_type_q = lambda e1, e2, t: "{z}\n\nQuestion: Do {x} and {y} interact with each other based on the text above?\n\nAnswer:".format(x=e1, y=e2, z=t)
        # relation_type_q = lambda e1, e2, t: "{z}\n\nQuestion: Do {x} and {y} interact with each other in the text above?\n\nAnswer:".format(x=e1, y=e2, z=t)
        
        
        relation_type_a = lambda answer: " {x}\n\n".format(x=answer)
        
        return {'relation_type_q': relation_type_q,
                'relation_type_a': relation_type_a}