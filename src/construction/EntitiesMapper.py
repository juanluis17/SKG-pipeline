from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON
from django.utils.encoding import smart_str
from sentence_transformers import util
from multiprocessing import Process
from urllib.parse import unquote
from random import shuffle
import networkx as nx
import pandas as pd
import numpy as np
import rdflib
import urllib
import random
import pickle
import torch
import json
import time
import csv
import os
from threading import RLock, Thread


class EntitiesMapper:
    def __init__(self, entities, all_pairs):
        self.entities = entities
        self.e2cso = {}
        self.e2wikidata = {}
        self.e2dbpedia = {}

        self.e2cso_processed = set()
        self.e2wikidata_processed = set()
        self.e2dbpedia_processed = set()

        self.e2cso_prev_len = 0
        self.e2wikidata_prev_len = 0
        self.e2dbpedia_prev_len = 0

        self.e2alternativeLabels = {}
        self.all_pairs = all_pairs
        self.e2neighbors = {}

        self.cso_map = {}
        self.dbpedia_map = {}
        self.emb_map = {}
        self.g = nx.Graph()

        self.csoResourcePath = '../../resources/CSO.3.1.csv'
        self.mappedTriples = {}  # main output of this class
        self.lock_cso = RLock()
        self.lock_dbpedia = RLock()
        self.lock_wiki = RLock()

    def linkThroughCSO(self, name, entities_to_explore):
        print('[{}] >> Mapping with cso started'.format(name))
        timepoint = time.time()
        entities_to_explore_subset = entities_to_explore
        if len(entities_to_explore_subset) <= 0:
            return

        print('[{}]\t >> Entities to be linked to cso: {}'.format(name, len(entities_to_explore_subset)))
        cso = rdflib.Graph()

        with open(self.csoResourcePath, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for s, p, o in csv_reader:
                s = s[1:-1]
                p = p[1:-1]
                o = o[1:-1]

                entity = s.replace('https://cso.kmi.open.ac.uk/topics/', '').replace('_', ' ')
                with self.lock_cso:
                    bool_cond = entity not in self.e2cso_processed
                if bool_cond:
                    print('[{}]\t >> Processing entity: {}'.format(name, entity))
                    with self.lock_cso:
                        self.e2cso_processed.add(entity)
                    if entity in entities_to_explore_subset:
                        with self.lock_cso:
                            self.e2cso[entity] = s
                        if p == 'http://www.w3.org/2002/07/owl#sameAs':
                            with self.lock_wiki:
                                if 'wikidata' in o:
                                    self.e2wikidata[entity] = o
                            with self.lock_dbpedia:
                                if 'dbpedia' in o:
                                    self.e2dbpedia[entity] = o

                entity = o.replace('https://cso.kmi.open.ac.uk/topics/', '').replace('_', ' ')
                with self.lock_cso:
                    bool_cond = entity not in self.e2cso_processed
                if bool_cond:
                    print('[{}]\t >> Processing entity: {}'.format(name, entity))
                    with self.lock_cso:
                        self.e2cso_processed.add(entity)
                    with self.lock_cso:
                        if entity in self.entities:
                            self.e2cso[entity] = o
                with self.lock_cso:
                    print('[{}]\t >> Mapped: {}'.format(name, len(self.e2cso)))
                    print('[{}]\t >> Processed: {}'.format(name, len(self.e2cso_processed)))
                    if ((len(self.e2cso.keys()) % 50) == 0) and self.e2cso_prev_len != len(self.e2cso.keys()):
                        print('[{}]\t >> CSO Processed'.format(name), len(self.e2cso),
                              'entities in {:.2f} secs.'.format(time.time() - timepoint))
                        pickle.dump(self.e2cso, open("../../resources/e2cso.pickle", "wb+"))
                        print('[{}]\t >> Saving Processed'.format(name), len(self.e2cso_processed))
                        pickle.dump(self.e2cso_processed, open("../../resources/e2cso_processed.pickle", "wb+"))
                        self.e2cso_prev_len = len(self.e2cso.keys())
                    if ((len(self.e2cso_processed) % 50) == 0):
                        print('[{}]\t >> Saving Processed'.format(name), len(self.e2cso_processed))
                        pickle.dump(self.e2cso_processed, open("../../resources/e2cso_processed.pickle", "wb+"))

        with self.lock_cso:
            print('[{}] > Saving...'.format(name))
            for e in entities_to_explore_subset:
                self.e2cso_processed.add(e)
            pickle.dump(self.e2cso, open("../../resources/e2cso.pickle", "wb+"))
            pickle.dump(self.e2cso_processed, open("../../resources/e2cso_processed.pickle", "wb+"))
            print('[{}] >> Mapped to CSO:'.format(name), len(self.e2cso))

    def linkThroughWikidata(self, name, entities_to_explore):
        print('[{}] >> Mapping with wikidata started'.format(name))
        timepoint = time.time()
        # sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        entities_to_explore_subset = entities_to_explore
        if len(entities_to_explore_subset) <= 0:
            return

        # print('sorting entities...')
        entities_to_explore = sorted(entities_to_explore_subset, key=lambda x: len(x), reverse=True)
        # print('sorted')
        c = 0
        print('[{}]\t >> Entities to be linked to wikidata: {}'.format(name, len(entities_to_explore)))
        while c < len(entities_to_explore):
            e = entities_to_explore[c]
            print('[{}]\t >> Processing entity: {}'.format(name, e))
            query = """
					SELECT DISTINCT ?entity ?altLabel
					WHERE{
							{
								?entity  <http://www.w3.org/2000/01/rdf-schema#label> \"""" + e + """\"@en .
								{?entity wdt:31+ wd:Q21198 }  UNION 
								{?entity wdt:31/wdt:P279* wd:Q21198} UNION
								{?entity wdt:P279+ wd:Q21198} UNION
								{?entity  wdt:P361+ wd:Q21198} UNION
								{ ?entity  wdt:P1269+ wd:Q21198} UNION
								{FILTER NOT EXISTS {?entity <http://schema.org/description> "Wikimedia disambiguation page"@en}}
								 OPTIONAL {
									?entity <http://www.w3.org/2004/02/skos/core#altLabel> ?altLabel .
									FILTER(LANG(?altLabel) = 'en')
								}
								

							}  UNION {
								?entity <http://www.w3.org/2004/02/skos/core#altLabel> \"""" + e + """\"@en .
								{?entity wdt:31+ wd:Q21198 }  UNION 
								{?entity wdt:31/wdt:P279* wd:Q21198} UNION
								{?entity wdt:P279+ wd:Q21198} UNION
								{?entity  wdt:P361+ wd:Q21198} UNION
								{ ?entity  wdt:P1269+ wd:Q21198} 
								 OPTIONAL {
									?entity <http://www.w3.org/2004/02/skos/core#altLabel> ?altLabel .
									FILTER(LANG(?altLabel) = 'en')
								}
								SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }

							}
						}
					"""

            url = 'https://query.wikidata.org/sparql'
            data = urllib.parse.urlencode({'query': query}).encode()
            headers = {"Accept": "application/sparql-results+json"}
            # print(e)
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                response = urllib.request.urlopen(req)

                if response.status == 200:

                    result = response.read().decode('ascii', errors='ignore')
                    jresponse = json.loads(result)
                    variables = jresponse['head']['vars']
                    for binding in jresponse['results']['bindings']:
                        with self.lock_wiki:
                            if 'entity' in binding and e not in self.e2wikidata:
                                if 'http://www.wikidata.org/entity/Q' in binding['entity']['value']:  # no wikidata:PXYZ
                                    self.e2wikidata[e] = binding['entity']['value']
                            # print('>    ', binding['entity']['value'])

                            if 'altLabel' in binding:
                                if binding['altLabel']['value'].lower() in entities_to_explore and binding['altLabel'][
                                    'value'].lower() not in self.e2wikidata:
                                    if 'http://www.wikidata.org/entity/Q' in binding['entity']['value']:
                                        self.e2wikidata[binding['altLabel']['value'].lower()] = binding['entity'][
                                            'value']
                # print('>    alt', binding['altLabel']['value'].lower(), binding['entity']['value'])
                with self.lock_wiki:
                    self.e2wikidata_processed.add(e)
                c += 1
                with self.lock_wiki:
                    print('[{}]\t >> Mapped: {}'.format(name, len(self.e2wikidata)))
                    print('[{}]\t >> Processed: {}'.format(name, len(self.e2wikidata_processed)))
                    if ((len(self.e2wikidata.keys()) % 50) == 0) and self.e2wikidata_prev_len != len(
                            self.e2wikidata.keys()):
                        print('[{}]\t >> Wikidata Processed'.format(name), len(self.e2wikidata),
                              'entities in {:.2f} secs.'.format(time.time() - timepoint))
                        pickle.dump(self.e2wikidata, open("../../resources/e2wikidata.pickle", "wb+"))
                        print('[{}]\t >> Wikidata Processed'.format(name), len(self.e2wikidata_processed))
                        pickle.dump(self.e2wikidata_processed,
                                    open("../../resources/e2wikidata_processed.pickle", "wb+"))
                        self.e2wikidata_prev_len = len(self.e2wikidata.keys())
                    if (len(self.e2wikidata_processed) % 50) == 0:
                        print('[{}]\t >> Wikidata Processed'.format(name), len(self.e2wikidata_processed))
                        pickle.dump(self.e2wikidata_processed,
                                    open("../../resources/e2wikidata_processed.pickle", "wb+"))

            # print('- \t >> Saving', len(self.e2wikidata), 'mappings')
            # raise urllib.error.HTTPError(req.full_url, '100', 'ciao', {'pippo':'pippo'}, fp=None)

            except urllib.error.HTTPError as err:
                # Return code error (e.g. 404, 501, ...)
                # print('Error 409', response.headers)
                print(err)
                # print('HTTPError: {}'.format(ex.code))
                print(err.headers)
                print('[{}] sleeping...'.format(name))
                time.sleep(60)
            except Exception as ex:
                print("[{}] {}".format(name, ex))
        with self.lock_wiki:
            print('[{}] Saving...'.format(name))
            pickle.dump(self.e2wikidata, open("../../resources/e2wikidata.pickle", "wb+"))
            pickle.dump(self.e2wikidata_processed, open("../../resources/e2wikidata_processed.pickle", "wb+"))
            print('[{}] Mapped to Wikidata:'.format(name), len(self.e2wikidata))

    def findNeiighbors(self):

        for s, o in self.all_pairs:
            if s not in self.e2neighbors:
                self.e2neighbors[s] = []
            if len(self.e2neighbors[s]) < 20:
                self.e2neighbors[s] += [o]

            if o not in self.e2neighbors:
                self.e2neighbors[o] = []
            if len(self.e2neighbors[o]) < 20:
                self.e2neighbors[o] += [s]

        '''eid = 0
        for s,o in self.all_pairs:
            if s not in self.e2id and (s in self.entities or o in self.entities):
                self.e2id[s] = eid
                self.id2e[eid] = s
                eid += 1

            if o not in self.e2id and (s in self.entities or o in self.entities):
                self.e2id[o] = eid
                self.id2e[eid] = o
                eid += 1

            if s in self.entities or o in self.entities:
                self.g.add_edge(self.e2id[s], self.e2id[o])
        '''

    def linkThroughDBpediaSpotLight(self, name, entities_to_explore):
        print('[{}] >> Mapping with dbpedia started'.format(name))
        entities_to_explore_subset = entities_to_explore
        if len(entities_to_explore_subset) <= 0:
            return
        print('[{}]\t >> Entities to be linked to dbpedia: {}'.format(name, len(entities_to_explore_subset)))
        self.findNeiighbors()

        c = 0
        timepoint = time.time()
        for e in entities_to_explore_subset:
            print('[{}]\t >> Processing entity: {}'.format(name, e))
            with self.lock_dbpedia:
                bool_cond = e not in self.e2dbpedia
            if bool_cond:
                neighbors = self.e2neighbors[e]

                # content = [e] + [self.id2e[nid] for nid in neighbors_ids[:20]]
                content = [e] + neighbors
                shuffle(content)
                content = ' '.join(content)

                url = 'https://api.dbpedia-spotlight.org/en/annotate'
                data = urllib.parse.urlencode({'text': content})
                headers = {"Accept": "application/json"}

                try:
                    req = urllib.request.Request(url + '?' + data, headers=headers)
                    response = urllib.request.urlopen(req)
                    if response.status == 200:

                        result = response.read().decode('ascii', errors='ignore')
                        jresponse = json.loads(result)

                        if 'Resources' in jresponse:
                            for resource in jresponse['Resources']:
                                if resource['@surfaceForm'] == e and float(resource['@similarityScore']) >= 0.8:
                                    with self.lock_dbpedia:
                                        self.e2dbpedia[e] = resource['@URI']
                                    break
                    with self.lock_dbpedia:
                        self.e2dbpedia_processed.add(e)
                except urllib.error.HTTPError as e:
                    print('[{}] HTTPError: {}'.format(name, e.code), 'sleeping...')
                    time.sleep(60)
                except:
                    print('[{}] E:'.format(name), e)
                    pass

                c += 1
                with self.lock_dbpedia:
                    print('[{}]\t >> Mapped: {}'.format(name, len(self.e2dbpedia)))
                    print('[{}]\t >> Processed: {}'.format(name, len(self.e2dbpedia_processed)))
                    if ((len(self.e2dbpedia.keys()) % 50) == 0) and self.e2dbpedia_prev_len != len(
                            self.e2dbpedia.keys()):
                        print('[{}] \t>> DBpedia Processed'.format(name), len(self.e2dbpedia), 'entities in',
                              (time.time() - timepoint),
                              'secs')
                        pickle.dump(self.e2dbpedia, open("../../resources/e2dbpedia.pickle", "wb+"))
                        print('[{}] \t>> DBpedia Processed'.format(name), len(self.e2dbpedia_processed))
                        pickle.dump(self.e2dbpedia_processed, open("../../resources/e2dbpedia_processed.pickle", "wb+"))
                        self.e2dbpedia_prev_len = len(self.e2dbpedia.keys())
                    if (len(self.e2dbpedia_processed) % 50) == 0:
                        print('[{}]\t >> Wikidata Processed'.format(name), len(self.e2dbpedia_processed))
                        pickle.dump(self.e2dbpedia_processed,
                                    open("../../resources/e2wikidata_processed.pickle", "wb+"))
        with self.lock_dbpedia:
            print('[{}] > Saving...'.format(name))
            pickle.dump(self.e2dbpedia, open("../../resources/e2dbpedia.pickle", "wb+"))
            pickle.dump(self.e2dbpedia_processed, open("../../resources/e2dbpedia_processed.pickle", "wb+"))
            print('[{}] >> Mapped to DBpedia:'.format(name), len(self.e2dbpedia))

    '''def save(self):

        pickle_out = open("../../resources/e2cso.pickle","wb")
        pickle.dump(self.e2cso, pickle_out)
        pickle_out.close()

        pickle_out = open("../../resources/e2wikidata.pickle","wb")
        pickle.dump(self.e2wikidata, pickle_out)
        pickle_out.close()

        pickle_out = open("../../resources/e2dbpedia.pickle","wb")
        pickle.dump(self.e2dbpedia, pickle_out)
        pickle_out.close()

        pickle_out = open("../../resources/e2alternativeLabels.pickle","wb")
        pickle.dump(self.e2alternativeLabels, pickle_out)
        pickle_out.close()
    '''

    def load(self):
        with self.lock_cso:
            if os.path.exists("../../resources/e2cso.pickle"):
                self.e2cso = pickle.load(open("../../resources/e2cso.pickle", "rb"))
                print('- Entities mapped with CSO:', len(self.e2cso))
            if os.path.exists("../../resources/e2cso_processed.pickle"):
                self.e2cso_processed = pickle.load(open("../../resources/e2cso_processed.pickle", "rb"))
                print('- Entities processed with CSO:', len(self.e2cso_processed))
        union_1 = set(self.e2cso_processed).union(set(self.e2cso.keys()))
        entities_to_explore = list(set(self.entities).difference(union_1))
        print('- Entities to explore with CSO:', len(entities_to_explore))
        threads_cso = []
        if len(entities_to_explore) != 0:
            chunk_size = int(len(entities_to_explore) / 1) + 1
            list_chunked = [list(entities_to_explore)[i:i + chunk_size] for i in
                            range(0, len(list(entities_to_explore)), chunk_size)]
            for i, chunk in enumerate(list_chunked):
                threads_cso.append(Thread(target=self.linkThroughCSO, args=("cso_{}".format(i), chunk,)))
        for th in threads_cso:
            th.start()

        with self.lock_dbpedia:
            if os.path.exists("../../resources/e2dbpedia.pickle"):
                self.e2dbpedia = pickle.load(open("../../resources/e2dbpedia.pickle", "rb"))
                print('- Entities mapped with DBPedia:', len(self.e2dbpedia))
            if os.path.exists("../../resources/e2dbpedia_processed.pickle"):
                self.e2dbpedia_processed = pickle.load(open("../../resources/e2dbpedia_processed.pickle", "rb"))
                print('- Entities processed with DBPedia:', len(self.e2dbpedia_processed))
        union_1 = set(self.e2dbpedia_processed).union(set(self.e2dbpedia.keys()))
        entities_to_explore = list(set(self.entities).difference(union_1))
        print('- Entities to explore with DBPedia:', len(entities_to_explore))
        threads_dbpedia = []
        if len(entities_to_explore) != 0:
            chunk_size = int(len(entities_to_explore) / 100) + 1
            list_chunked = [list(entities_to_explore)[i:i + chunk_size] for i in
                            range(0, len(list(entities_to_explore)), chunk_size)]
            for i, chunk in enumerate(list_chunked):
                threads_dbpedia.append(
                    Thread(target=self.linkThroughDBpediaSpotLight, args=("DBpedia_{}".format(i), chunk,)))
        for th in threads_dbpedia:
            th.start()

        with self.lock_wiki:
            if os.path.exists("../../resources/e2wikidata.pickle"):
                self.e2wikidata = pickle.load(open("../../resources/e2wikidata.pickle", "rb"))
                print('- Entities mapped with e2wikidata:', len(self.e2wikidata))
            if os.path.exists("../../resources/e2wikidata_processed.pickle"):
                self.e2wikidata_processed = pickle.load(open("../../resources/e2wikidata_processed.pickle", "rb"))
                print('- Entities processed with e2wikidata:', len(self.e2wikidata_processed))
        union_1 = set(self.e2wikidata_processed).union(set(self.e2wikidata.keys()))
        entities_to_explore = list(set(self.entities).difference(union_1))
        print('- Entities to explore with e2wikidata:', len(entities_to_explore))
        threads_wiki = []
        if len(entities_to_explore) != 0:
            chunk_size = int(len(entities_to_explore) / 50) + 1
            list_chunked = [list(entities_to_explore)[i:i + chunk_size] for i in
                            range(0, len(list(entities_to_explore)), chunk_size)]
            for i, chunk in enumerate(list_chunked):
                threads_wiki.append(Thread(target=self.linkThroughWikidata, args=("wiki_{}".format(i), chunk,)))
        for th in threads_wiki:
            th.start()

        try:
            for th in threads_cso:
                th.join()
            for th in threads_dbpedia:
                th.join()
            for th in threads_wiki:
                th.join()
        except Exception as e:
            print(e)

    def run(self):
        print('\t>> Entities to be mapped:', len(self.entities))
        self.load()

    def getMaps(self):
        return self.e2cso, self.e2dbpedia, self.e2wikidata


if __name__ == '__main__':
    entities = ['neural network', 'artificial neural network', 'ann', 'computer science', 'ontology alignment',
                'convolutional neural network', 'ontology matching', 'neural network', 'cnn', 'deep learning',
                'precision', 'recall', 'ontology', 'semantic web']
    triples = [('cnn', 'convolutional neural network'), \
               ('deep learning', 'convolutional neural network'), \
               ('deep learning', 'computer science'), \
               ('deep learning', 'recall'), \
               ('precision', 'recall'), \
               ('machine learning', 'deep learning'), \
               ('ontology alignment', 'ontology'), \
               ('ontology alignment', 'ontology matching'), \
               ('ontology', 'semantic web'), \
               ('ontology alignment', 'ontology'), \
               ('ontology', 'computer science'), \
               ('neural network', 'artificial neural network'), \
               ('machine learning', 'ann'), \
               ('cnn', 'ann'), \
               ('cnn', 'semantic web') \
               ]
    mapper = EntitiesMapper(entities, triples)
    mapper.run()
    print()