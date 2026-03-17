from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict
from tqdm import tqdm
import pandas as pd

from bioel.logger import setup_logger
from bioel.utils.bigbio_utils import dataset_unique_tax_ids
from bioel.utils.snomed_ct_it_utils import Snomed

import obonet
import csv
from bioel.utils.obo_utils import _obo_extract_definition, _obo_extract_synonyms
from bioel.utils.umls_utils import UmlsMappings
import warnings
import ujson
import json
from bioel.utils.file_cache import get_path

logger = setup_logger()


@dataclass
class BiomedicalEntity:
    """
    Class for keeping track of all relevant fields in an ontology
    """

    cui: str
    name: str
    types: List[str]
    aliases: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    equivalant_cuis: Optional[List[str]] = None
    taxonomy: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class BiomedicalOntology:
    name: str
    types: List[str] = field(default_factory=list)
    entities: Dict[str, BiomedicalEntity] = field(
        default_factory=dict
    )  # Dict mapping CUI: BiomedicalEntity
    abbrev: Optional[str] = None  # Abbreviated name of ontology if different than name
    metadata: Optional[dict] = None
    dataset: Optional[str] = (
        None  # Name of the dataset to work with (only for Entrez ontology)
    )

    def get_canonical_name(self):
        """
        Get name of entities in the ontology
        """
        canonical_names = {cui: entity.name for cui, entity in self.entities.items()}
        return canonical_names

    def get_aliases(self):
        """
        Get aliases of entities in the ontology
        """
        aliases_dict = {cui: entity.aliases for cui, entity in self.entities.items()}
        return aliases_dict

    def get_definition(self):
        """
        Get definition of entities in the ontology
        """
        definitions_dict = {
            entity.cui: entity.definition
            for entity in self.entities.values()
            if entity.definition is not None
        }
        return definitions_dict

    def get_types(self):
        """
        Get types of entities in the ontology
        """
        types_dict = {cui: entity.types for cui, entity in self.entities.items()}
        return types_dict

    @classmethod
    def load_obo(
        cls, filepath, name=None, prefix_to_keep=None, entity_type=None, abbrev=None
    ):
        """
        Read an ontology in .obo format

        Parameters:
        ----------------------
            filepath: str (optional)
                Path to .obo formatted ontology.  Can be a URL.
            prefix_to_keep: str (optional)
        """

        entities = {}
        if entity_type:
            types = [entity_type]
        else:
            types = []

        logger.info(f"Reading OBO ontology from {filepath}")
        ontology = obonet.read_obo(filepath)

        data_keys = set([])
        for curie, data in tqdm(ontology.nodes(data=True)):
            # Exclude CUIs that are from cross-referenced ontologies
            data_keys.update(set(data.keys()))

            if prefix_to_keep is not None:
                if not curie.startswith(prefix_to_keep):
                    continue

            # Extract name and synonyms
            if "name" not in data:
                synonyms = _obo_extract_synonyms(data)
            else:
                synonyms = [data["name"]] + _obo_extract_synonyms(data)

            # Include deprecated CUIs as alternative ids
            alt_cuis = None
            if "alt_id" in data and len(data["alt_id"]) > 0:
                if prefix_to_keep:
                    alt_cuis = [
                        x for x in data["alt_id"] if x.startswith(prefix_to_keep)
                    ]
                else:
                    alt_cuis = data["alt_id"]

            # Skip entities with no name/aliases
            if len(synonyms) == 0:
                logger.warning(
                    f"Data entry for CUI {curie} has no listed names/aliases.  Skipping."
                )
                continue
            ent_name = synonyms[0]
            if len(synonyms) > 1:
                other_synonyms = synonyms[1:]
            else:
                other_synonyms = []

            # Get definition if it exists
            definition = _obo_extract_definition(data)

            ent = BiomedicalEntity(
                cui=curie,
                name=ent_name,
                aliases=other_synonyms,
                types=types,
                definition=definition,
                equivalant_cuis=alt_cuis,
            )
            if curie in entities:
                logger.warning(f"Duplicate CUI {curie} found in ontology.  Skipping.")
                continue

            entities[curie] = ent

        if not name:
            if filepath.startswith("http"):
                parsed_name = filepath.split("/")[-1].split(".")[0]
                logger.warning(
                    f"No ontology name provided.  Using name from provided URL: {parsed_name}"
                )
                name = parsed_name

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)

    @classmethod
    def load_medic(cls, filepath, name=None, abbrev=None, api_key=""):
        """
        Read medic ontology

        Parameters:
        ----------------------
            filepath: str (Pointing to the medic directory)
            name: str (optional)
            abbrev: str (optional)
            api_key: str (optional)
        """
        entities = {}
        types = []

        logger.info(f"Reading entrez from {filepath}")
        col_names = "Disease_Name   Disease_ID       Alt_Disease_IDs   Definition      Parent_IDs       Tree_Numbers     Parent_Tree_Numbers       Synonyms        Slim_Mappings".split()

        # Open the TSV file
        medic = pd.read_csv(
            filepath,
            comment="#",
            delimiter="\t",
            na_filter=False,
            names=col_names,
            usecols=[
                "Disease_Name",
                "Disease_ID",
                "Alt_Disease_IDs",
                "Definition",
                "Parent_IDs",
                "Tree_Numbers",
                "Parent_Tree_Numbers",
                "Synonyms",
                "Slim_Mappings",
            ],
        )
        for index, row in medic.iterrows():
            equivalant_cuis = [row["Disease_ID"]]
            alt_ids = (
                row["Alt_Disease_IDs"].split("|") if row["Alt_Disease_IDs"] else []
            )
            for alt_id in alt_ids:
                if alt_id not in equivalant_cuis and alt_id[:2] != "DO":
                    equivalant_cuis.append(alt_id)

            entity = BiomedicalEntity(
                cui=row["Disease_ID"],
                name=row["Disease_Name"],
                types="Disease",
                aliases=row["Synonyms"],
                definition=row["Definition"],
                equivalant_cuis=equivalant_cuis,
            )

            if row["Disease_ID"] in entities:
                logger.warning(
                    f"Duplicate CUI {row['Disease_ID']} found in ontology.  Skipping."
                )
                continue

            entities[row["Disease_ID"]] = entity

            types.append("Disease")
        return cls(entities=entities, types=types, name=name, abbrev=abbrev)

    @classmethod
    def load_entrez(cls, filepath, dataset, name=None, abbrev=None, api_key=""):
        """
        Read medic ontology

        Parameters:
        ----------------------
            filepath: str (Pointing to the medic directory)
            name: str (optional)
            abbrev: str (optional)
            api_key: str (optional)
        """

        entities = {}
        types = []

        logger.info(f"Reading entrez from {filepath}")

        # Open the TSV file
        entrez = pd.read_csv(
            filepath,
            delimiter="\t",
            usecols=[
                "#tax_id",
                "GeneID",
                "Symbol",
                "Synonyms",
                "Symbol_from_nomenclature_authority",
                "Full_name_from_nomenclature_authority",
                "Other_designations",
                "type_of_gene",
                "description",
                "dbXrefs",
            ],
            na_filter=False,
            low_memory=False,
        ).rename(
            {
                "Symbol_from_nomenclature_authority": "official_symbol",
                "Full_name_from_nomenclature_authority": "official_name",
                "#tax_id": "tax_id",
            },
            axis=1,
        )
        entrez.columns = [x.lower() for x in entrez.columns]

        unique_tax_ids = dataset_unique_tax_ids(dataset, entrez)

        geneid_mask = (
            (entrez.tax_id.isin(unique_tax_ids))
            & (~entrez.type_of_gene.isin(["unknown", "tRNA", "biological-region"]))
            & (entrez.description != "hypothetical protein")
            & (~entrez.official_name.map(lambda x: x.lower().startswith("predicted")))
        )
        entrez = entrez[geneid_mask]

        entrez.replace("-", "", inplace=True)
        entrez["geneid"] = entrez["geneid"].map(lambda x: f"NCBIGene:{x}")
        entrez["metadata"] = entrez[
            [
                "official_symbol",
                "official_name",
            ]
        ].progress_apply(
            lambda x: ";".join(list(set([i for i in x if i.strip() != "-"]))), axis=1
        )

        for index, row in entrez.iterrows():

            entity = BiomedicalEntity(
                cui=row["geneid"],
                name=row["symbol"],
                types=row["type_of_gene"],
                aliases=row["synonyms"],
                definition=row["description"],
                taxonomy=row["tax_id"],
                metadata=row["metadata"],
            )

            if row["geneid"] in entities:
                logger.warning(
                    f"Duplicate CUI {row['geneid']} found in ontology.  Skipping."
                )
                continue

            entities[row["geneid"]] = entity

            types.append(row["type_of_gene"])

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)
    
    @classmethod
    def load_umls(
        cls, filepath, path_st21pv_cui=None, name=None, abbrev=None, api_key=""
    ):
        """
        Read an ontology from the UMLS Directory

        Parameters:
        ----------------------
            filepath: str (Pointing to the UMLS directory)
            path_st21pv_cui : str (optional : Path to st21pv cuis subset for umls)
            name: str (optional)
            abbrev: str (optional)
            api_key: str (optional)
        """

        entities = {}
        types = []

        logger.info(f"Reading UMLS from {filepath}")
        umls = UmlsMappings(umls_dir=filepath, umls_api_key=api_key)

        # Get the Canonial Names
        lowercase = False
        umls_to_name = umls.get_canonical_name(
            ontologies_to_include="all",
            use_umls_curies=True,
            lowercase=lowercase,
        )

        # Group by the canonical names to group the alias and types
        all_umls_df = (
            umls.umls.query('lang == "ENG"')
            .groupby("cui")
            .agg(
                {
                    "alias": lambda x: list(set(x)),
                    "tui": "first",
                    "group": "first",
                    "def": "first",
                }
            )
            .reset_index()
        )

        if path_st21pv_cui:
            with open(path_st21pv_cui, "r") as file:
                umls_cuis_st21pv = ujson.load(file)
            all_umls_df = all_umls_df[
                all_umls_df["cui"].isin(umls_cuis_st21pv)
            ].reset_index()

        all_umls_df["name"] = all_umls_df.cui.map(umls_to_name)
        all_umls_df["alias"] = all_umls_df[["name", "alias"]].apply(
            lambda x: list(set(x[1]) - set([x[0]])), axis=1
        )
        all_umls_df["cui"] = all_umls_df["cui"].map(lambda x: "UMLS:" + x)
        all_umls_df["has_definition"] = all_umls_df["def"].map(lambda x: x is not None)
        all_umls_df["num_aliases"] = all_umls_df["alias"].map(lambda x: len(x))

        print("Number of entities :", all_umls_df.shape[0])

        for index, row in tqdm(all_umls_df.iterrows(), desc="Loading UMLS Ontology"):
            entity = BiomedicalEntity(
                cui=row["cui"],
                name=row["name"],
                types=row["tui"],
                aliases=row["alias"],
                definition=row["def"],
                metadata={
                    "group": row["group"],
                },
            )
            if row["cui"] in entities:
                logger.warning(
                    f"Duplicate CUI {row['cui']} found in ontology.  Skipping."
                )
                continue

            entities[row["cui"]] = entity
            types.append(row["tui"])

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)

    @classmethod
    def load_entrez(cls, filepath, dataset, name=None, abbrev=None, api_key=""):
        """
        Read medic ontology

        Parameters:
        ----------------------
            filepath: str (Pointing to the medic directory)
            name: str (optional)
            abbrev: str (optional)
            api_key: str (optional)
        """

        entities = {}
        types = []

        logger.info(f"Reading entrez from {filepath}")

        # Open the TSV file
        entrez = pd.read_csv(
            filepath,
            delimiter="\t",
            usecols=[
                "#tax_id",
                "GeneID",
                "Symbol",
                "Synonyms",
                "Symbol_from_nomenclature_authority",
                "Full_name_from_nomenclature_authority",
                "Other_designations",
                "type_of_gene",
                "description",
                "dbXrefs",
            ],
            na_filter=False,
            low_memory=False,
        ).rename(
            {
                "Symbol_from_nomenclature_authority": "official_symbol",
                "Full_name_from_nomenclature_authority": "official_name",
                "#tax_id": "tax_id",
            },
            axis=1,
        )
        entrez.columns = [x.lower() for x in entrez.columns]

        unique_tax_ids = dataset_unique_tax_ids(dataset, entrez)

        geneid_mask = (
            (entrez.tax_id.isin(unique_tax_ids))
            & (~entrez.type_of_gene.isin(["unknown", "tRNA", "biological-region"]))
            & (entrez.description != "hypothetical protein")
            & (~entrez.official_name.map(lambda x: x.lower().startswith("predicted")))
        )
        entrez = entrez[geneid_mask]

        entrez.replace("-", "", inplace=True)
        entrez["geneid"] = entrez["geneid"].map(lambda x: f"NCBIGene:{x}")
        entrez["metadata"] = entrez[
            [
                "official_symbol",
                "official_name",
            ]
        ].progress_apply(
            lambda x: ";".join(list(set([i for i in x if i.strip() != "-"]))), axis=1
        )

        for index, row in entrez.iterrows():

            entity = BiomedicalEntity(
                cui=row["geneid"],
                name=row["symbol"],
                types=row["type_of_gene"],
                aliases=row["synonyms"],
                definition=row["description"],
                taxonomy=row["tax_id"],
                metadata=row["metadata"],
            )

            if row["geneid"] in entities:
                logger.warning(
                    f"Duplicate CUI {row['geneid']} found in ontology.  Skipping."
                )
                continue

            entities[row["geneid"]] = entity

            types.append(row["type_of_gene"])

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)

    @classmethod
    def load_mesh(cls, filepath, name=None, abbrev=None, api_key=""):
        """
        Read an ontology from the MESH Directory

        Parameters:
        ----------------------
            filepath: str (Pointing to the UMLS directory)
            name: str (optional)
            abbrev: str (optional)
            api_key: str (optional)
        """
        entities = {}
        types = []

        logger.info(f"Reading MESH from : {filepath}")
        umls = UmlsMappings(umls_dir=filepath, umls_api_key=api_key)

        # Get the Canonial Names
        lowercase = False
        mesh_to_name = umls.get_canonical_name(
            ontologies_to_include=["MSH"],
            use_umls_curies=False,
            mapping_cols={"MSH": "sdui"},
            prefixes={"MSH": "MESH"},
            lowercase=lowercase,
        )

        mesh_to_alias = umls.get_aliases(
            ontologies_to_include=["MSH"],
            use_umls_curies=False,
            mapping_cols={"MSH": "sdui"},
            prefixes={"MSH": "MESH"},
            lowercase=lowercase,
        )

        mesh_cui2definition = umls.get_definition(
            ontologies_to_include=["MSH"],
            use_umls_curies=False,
            mapping_cols={"MSH": "sdui"},
            prefixes={"MSH": "MESH"},
            lowercase=lowercase,
        )

        mesh_to_types, mesh_to_groups = umls.get_types_and_groups(
            ontologies_to_include=["MSH"],
            use_umls_curies=False,
            mapping_cols={"MSH": "sdui"},
            prefixes={"MSH": "MESH"},
            lowercase=lowercase,
        )
        i = 0
        for cui, _name in tqdm(mesh_to_name.items()):
            # ent_type = mesh_to_types[cui]
            ent_type = mesh_to_groups[cui][0]
            # if i < 1:
            #     print(f"{mesh_to_groups[cui][0]=}")
            #     print(f"{mesh_to_types[cui]=}")
            other_aliases = [x for x in mesh_to_alias[cui] if x != _name]
            joined_aliases = " ; ".join(other_aliases)
            entity = BiomedicalEntity(
                cui=cui,
                name=_name,
                types=ent_type,
                aliases=joined_aliases,
                definition=(
                    mesh_cui2definition[cui] if cui in mesh_cui2definition else None
                ),
                metadata={
                    "group": mesh_to_groups[cui],
                },
            )

            if cui in entities:
                logger.warning(f"Duplicate CUI {cui} found in ontology.  Skipping.")
                continue

            entities[cui] = entity
            types.append(ent_type)

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)
        
    @classmethod
    def load_json(cls, filepath, name=None):
        '''
        file_path: str, required.
            The file path to the json/jsonl representation of the KB to load.
        name: str, optional.
            The ontology name to load.
        '''
        if filepath is None:
            raise ValueError(
                "provide a valid path"
            )
        if filepath.endswith("jsonl"):
            raw = (json.loads(line) for line in open(get_path(filepath)))
        else:
            raw = json.load(open(get_path(filepath)))

        logger.info(f"Reading the given Json ontology from {filepath}")
        
        index = 0
        types = []
        entities = {}
        metadata ={}
        if name is None:
            name = "json_onto"
        for concept in raw:
            types.append(concept["types"])
            metadata[concept["concept_id"]] = index
            index += 1
            if concept["concept_id"] in entities:
                logger.warning(f"Duplicate CUI {concept['concept_id']} found in ontology.  Skipping.")
                continue
            elif 'definition' in concept:
                entities[concept["concept_id"]] = BiomedicalEntity(concept["concept_id"], concept["canonical_name"], concept["types"], concept["aliases"], concept["definition"])
            else:
                entities[concept["concept_id"]] = BiomedicalEntity(concept["concept_id"], concept["canonical_name"], concept["types"], concept["aliases"])
                
        return cls(entities = entities, types = types, metadata = metadata, name = name)


    @classmethod
    def load_snomed_ct_it(cls, filepath, release_id = "20251101", name=None, abbrev=None, api_key=""): # Work with COMETA and AAP datasets
        """
        Read an ontology from the SNOMED CT Directory
        
        Parameters:
        ----------------------
            filepath: str (Pointing to the UMLS directory)
            release_id: str (default to "20251101")
            name: str (optional)
            abbrev: str (optional)
            api_key: str (optional)
        """

        entities = {}
        types = []

        logger.info(f"Reading SNOMED CT from : {filepath}")
        snomed_path = filepath
        snomed = Snomed(snomed_path=snomed_path, release_id=release_id)
        snomed.load_snomed()

        # Get the Canonial Names
        for cui in snomed.index_concept : 
            concept = snomed.index_concept[cui][0]
            aliases = snomed.index_concept[cui][1:]
            definition = snomed.index_definition.get(cui, None)
            new_cui = "SNOMEDCT:" + cui
            entity = BiomedicalEntity(
                cui=new_cui,
                name=concept,
                aliases='|'.join(aliases) if aliases else '',
                definition=definition,
                types=None,
            )
            entities[new_cui] = entity

        return cls(entities=entities, types=types, name=name, abbrev=abbrev)

    def load_ncbi_taxon(self):
        pass

    def load_csv(self):
        pass

    def load_json(self):
        pass


@dataclass
class CompositeOntology:
    ontologies: dict  # Dict of each ontology used

    def load_from_config(self, config=None, config_path=None):
        pass
