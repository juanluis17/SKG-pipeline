from EntitiesValidator import EntitiesValidator
from RelationsManager import RelationsManager
from EntitiesCleaner import EntitiesCleaner
from EntitiesMapper import EntitiesMapper
from KGDataDumper import KGDataDumper
import pickle
import json
import os
import gc
import nltk

nltk.download('wordnet')
nltk.download('wordnet_ic')


class TriplesGenerator:
    def __init__(self, data_extracted_dir):
        self.entities2files = {}
        self.dygiepp2files = {}
        self.openie2files = {}
        self.pos2files = {}
        self.dependency2files = {}
        self.data_extracted_dir = data_extracted_dir  # '../../outputs/extracted_triples/'
        self.e2selected_type = {}
        self.e2cso = {}
        self.e2dbpedia = {}
        self.e2wikidata = {}

    ############ Data Loading #######################################################################################################

    def addDataInTripleDict(self, dic, triples_list, doc_key):
        for (s, p, o) in triples_list:
            if (s, p, o) not in dic:
                dic[(s, p, o)] = []
            dic[(s, p, o)] += [doc_key]

    def loadData(self):
        for filename in os.listdir(self.data_extracted_dir):
            # c = 0
            if filename[-5:] == '.json':
                f = open(self.data_extracted_dir + filename, 'r').readlines()
                for row in f:
                    try:
                        paper_data = json.loads(row.strip())
                        self.addDataInTripleDict(self.dygiepp2files, paper_data['dygiepp_triples'],
                                                 paper_data['doc_key'])
                        self.addDataInTripleDict(self.openie2files, paper_data['openie_triples'], paper_data['doc_key'])
                        self.addDataInTripleDict(self.pos2files, paper_data['pos_triples'], paper_data['doc_key'])
                        self.addDataInTripleDict(self.dependency2files, paper_data['dependency_triples'],
                                                 paper_data['doc_key'])
                        for (e, etype) in paper_data['entities']:
                            if (e, etype) not in self.entities2files:
                                self.entities2files[(e, etype)] = []
                            self.entities2files[(e, etype)] += [paper_data['doc_key']]
                    except:
                        pass

    ###################################################################################################################################

    ########### CLeaning of entities ##################################################################################################

    def applyCleanerMap(self, relations2files, cleaner_map):
        tool_triples2files = {}
        for (s, p, o), files in relations2files.items():
            if s in cleaner_map and o in cleaner_map:
                if (cleaner_map[s], p, cleaner_map[o]) in tool_triples2files:
                    tool_triples2files[(cleaner_map[s], p, cleaner_map[o])].update(set(files))
                else:
                    tool_triples2files[(cleaner_map[s], p, cleaner_map[o])] = set(files)
        return tool_triples2files

    def updateThroughCleanerMap(self, cleaner_map):
        tmp_entities2files = {}
        for (e, e_type), files in self.entities2files.items():
            if e in cleaner_map:
                if (cleaner_map[e], e_type) in tmp_entities2files:
                    tmp_entities2files[(cleaner_map[e], e_type)].update(set(files))
                else:
                    tmp_entities2files[(cleaner_map[e], e_type)] = set(files)
        self.entities2files = tmp_entities2files

        self.dygiepp2files = self.applyCleanerMap(self.dygiepp2files, cleaner_map)
        self.pos2files = self.applyCleanerMap(self.pos2files, cleaner_map)
        self.openie2files = self.applyCleanerMap(self.openie2files, cleaner_map)
        self.dependency2files = self.applyCleanerMap(self.dependency2files, cleaner_map)

    ###################################################################################################################################

    ############ Validation of entities ###############################################################################################

    def applyValidEntities(self, validEntities, relations2files):
        new_relations2files = {}
        for (s, p, o), files in relations2files.items():
            if s in validEntities and o in validEntities:
                if (s, p, o) in new_relations2files:
                    new_relations2files[(s, p, o)].update(set(files))
                else:
                    new_relations2files[(s, p, o)] = set(files)
        return new_relations2files

    def updateThroughValidEntities(self, validEntities):

        tmp_entities2files = {}
        for (e, e_type), files in self.entities2files.items():
            if e in validEntities:
                if (e, e_type) in tmp_entities2files:
                    tmp_entities2files[(e, e_type)].update(set(files))
                else:
                    tmp_entities2files[(e, e_type)] = set(files)
        self.entities2files = tmp_entities2files

        self.dygiepp2files = self.applyValidEntities(validEntities, self.dygiepp2files)
        self.openie2files = self.applyValidEntities(validEntities, self.openie2files)
        self.pos2files = self.applyValidEntities(validEntities, self.pos2files)
        self.dependency2files = self.applyValidEntities(validEntities, self.dependency2files)

    ###################################################################################################################################

    ########################################### Entities type and frequencies ##########################################################
    def entitiesTyping(self):
        self.e2types = {}
        for (e, e_type), files in self.entities2files.items():
            if e not in self.e2types:
                self.e2types[e] = {}

            if e_type != 'Generic':
                self.e2types[e][e_type] = len(files)
            else:
                if 'OtherScientificTerm' in self.e2types[e]:
                    self.e2types[e]['OtherScientificTerm'] += len(files)
                else:
                    self.e2types[e]['OtherScientificTerm'] = len(files)

        for e in self.e2types:
            occurence_count = self.e2types[e]

            # most frequent ignoring OtherEntity and CSOTopic
            selected_type = None
            max_freq = 0
            for etype, freq in dict(occurence_count).items():
                if etype != 'OtherScientificTerm' and etype != 'CSO Topic' and freq > max_freq:
                    selected_type = etype
                    max_freq = freq

            # if no Material, Method, etc. OtherEntity
            if selected_type == None:
                selected_type = 'OtherScientificTerm'

            self.e2selected_type[e] = selected_type

        with open('../../resources/e2selected_type.pickle', 'wb') as f:
            pickle.dump(self.e2selected_type, f)

    def entitiesFreq(self, cut_freq):
        e2count = {}
        for data_dict in [self.dygiepp_pair2info, self.openie_pair2info, self.pos_pair2info, self.dep_pair2info]:
            for (s, o) in data_dict:
                if s not in e2count: e2count[s] = 0
                if o not in e2count: e2count[o] = 0
                e2count[s] += len(data_dict[(s, o)])
                e2count[o] += len(data_dict[(s, o)])

        return [e for e, c in e2count.items() if c >= cut_freq]

    ###################################################################################################################################

    def createCheckpoint(self, name, els):
        with open('./ckpts/' + name + '.pickle', 'wb') as f:
            pickle.dump(els, f)

    def loadCheckpoint(self, name):
        with open('./ckpts/' + name + '.pickle', 'rb') as f:
            return pickle.load(f)

    def run(self):

        ckpts_loading = os.path.exists('./ckpts/loading.pickle')
        ckpts_cleaning = os.path.exists('./ckpts/cleaning.pickle')
        ckpts_validation = os.path.exists('./ckpts/validation.pickle')
        ckpts_mapping = os.path.exists('./ckpts/mapping.pickle')
        ckpts_relations_handler = os.path.exists('./ckpts/relations_handler.pickle')

        print('--------------------------------------')
        print('>> Loading')
        if ckpts_loading and not ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
            print('\t>> Loaded from ckpts')
            self.dygiepp2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files = self.loadCheckpoint(
                'loading')
            print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
            print(' \t- openie triples:\t', len(self.openie2files))
            print(' \t- pos triples:\t\t', len(self.pos2files))
            print(' \t- dep triples:\t\t', len(self.dependency2files))
        elif not ckpts_loading and not ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
            self.loadData()
            self.createCheckpoint('loading', (
                self.dygiepp2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files))
            print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
            print(' \t- openie triples:\t', len(self.openie2files))
            print(' \t- pos triples:\t\t', len(self.pos2files))
            print(' \t- dep triples:\t\t', len(self.dependency2files))
        else:
            print('\t>> skipped')
        print('--------------------------------------')

        print('>> Entity cleaning')
        if ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
            print('\t>> Loaded from ckpts')
            self.dygiepp2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files = self.loadCheckpoint(
                'cleaning')
            print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
            print(' \t- openie triples:\t', len(self.openie2files))
            print(' \t- pos triples:\t\t', len(self.pos2files))
            print(' \t- dep triples:\t\t', len(self.dependency2files))
        elif not ckpts_cleaning and not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
            ec = EntitiesCleaner(set([e for (e, e_type) in self.entities2files.keys()]))
            ec.run()
            cleaner_map = ec.get()
            self.createCheckpoint('entity2cleaned_entity', cleaner_map)
            self.updateThroughCleanerMap(cleaner_map)
            del cleaner_map
            gc.collect()
            self.createCheckpoint('cleaning', (
                self.dygiepp2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files))
            print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
            print(' \t- openie triples:\t', len(self.openie2files))
            print(' \t- pos triples:\t\t', len(self.pos2files))
            print(' \t- dep triples:\t\t', len(self.dependency2files))
        else:
            print('\t>> skipped')
        print('--------------------------------------')

        print('>> Entity validation')
        if ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
            print('\t>> Loaded from ckpts')
            self.dygiepp2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files = self.loadCheckpoint(
                'validation')
            print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
            print(' \t- openie triples:\t', len(self.openie2files))
            print(' \t- pos triples:\t\t', len(self.pos2files))
            print(' \t- dep triples:\t\t', len(self.dependency2files))
        elif not ckpts_validation and not ckpts_relations_handler and not ckpts_mapping:
            ev = EntitiesValidator(set([e for (e, e_type) in self.entities2files.keys()]))
            ev.run()
            valid_entities = ev.get()
            self.updateThroughValidEntities(valid_entities)
            del ev
            gc.collect()
            self.createCheckpoint('validation', (
                self.dygiepp2files, self.openie2files, self.pos2files, self.dependency2files, self.entities2files))
            print(' \t- dygiepp triples:\t', len(self.dygiepp2files))
            print(' \t- openie triples:\t', len(self.openie2files))
            print(' \t- pos triples:\t\t', len(self.pos2files))
            print(' \t- dep triples:\t\t', len(self.dependency2files))
        else:
            print('\t>> skipped')
        print('--------------------------------------')

        print('>> Relations handling')
        if ckpts_relations_handler:
            print('\t>> Loaded from ckpts')
            self.dygiepp_pair2info, self.openie_pair2info, self.pos_pair2info, self.dep_pair2info, self.entities2files = self.loadCheckpoint(
                'relations_handler')
            print(' \t- dygiepp pairs:\t', len(self.dygiepp_pair2info))
            print(' \t- openie pairs:\t\t', len(self.openie_pair2info))
            print(' \t- pos pairs:\t\t', len(self.pos_pair2info))
            print(' \t- dep pairs:\t\t', len(self.dep_pair2info))
        elif not ckpts_relations_handler and not ckpts_mapping:
            rm = RelationsManager(self.dygiepp2files, self.pos2files, self.openie2files, self.dependency2files)
            rm.run()
            self.dygiepp_pair2info, self.pos_pair2info, self.openie_pair2info, self.dep_pair2info = rm.get()
            del rm
            del self.dygiepp2files
            del self.openie2files
            del self.pos2files
            del self.dependency2files
            gc.collect()
            self.createCheckpoint('relations_handler', (
                self.dygiepp_pair2info, self.openie_pair2info, self.pos_pair2info, self.dep_pair2info,
                self.entities2files))
            print(' \t- dygiepp pairs:\t', len(self.dygiepp_pair2info))
            print(' \t- openie pairs:\t\t', len(self.openie_pair2info))
            print(' \t- pos pairs:\t\t', len(self.pos_pair2info))
            print(' \t- dep pairs:\t\t', len(self.dep_pair2info))
        else:
            print('\t>> skipped')
        print('--------------------------------------')

        print('>> Mapping to external resources')
        if ckpts_mapping:
            print('\t>> Loaded from ckpts')
            self.e2cso, self.e2dbpedia, self.e2wikidata = self.loadCheckpoint('mapping')
        elif not ckpts_mapping:
            all_pairs = set(self.dygiepp_pair2info.keys()) | set(self.pos_pair2info.keys()) | set(
                self.openie_pair2info.keys()) | set(self.dep_pair2info.keys())
            # mapper = EntitiesMapper([e for e, t in self.entities2files.keys()], all_pairs)
            cut_freq = 1
            mapper = EntitiesMapper(self.entitiesFreq(cut_freq), all_pairs)
            mapper.run()
            self.e2cso, self.e2dbpedia, self.e2wikidata = mapper.getMaps()
            del mapper
            gc.collect()
            self.createCheckpoint('mapping', (self.e2cso, self.e2dbpedia, self.e2wikidata))
        else:
            print('\t>> skipped')
        print('--------------------------------------')

        print('>> Data dumping and merging')
        self.entitiesTyping()
        dumper = KGDataDumper(self.dygiepp_pair2info, self.pos_pair2info, self.openie_pair2info, self.dep_pair2info,
                              self.e2cso, self.e2dbpedia, self.e2wikidata, self.e2selected_type)
        dumper.run()
        print('--------------------------------------')


import argparse


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_extracted_dir", type=str, default='../../outputs/extracted_triples/', help="")
    return parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists('./ckpts/'):
        os.makedirs('./ckpts/')
    args = get_args()
    data_extracted_dir = args.data_extracted_dir
    tg = TriplesGenerator(data_extracted_dir)
    tg.run()
