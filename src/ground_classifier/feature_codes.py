"""Feature (LAS classification) codes used by the input data and by the application.

Source point clouds can use any classification codes (e.g. 187, 210 and 225 may all
be wires). The application works with a small set of named features (ground, noise,
...) and writes one output code for each. ``FeatureCodes`` holds both vocabularies and
the mapping between them, and is the single place where code <-> name <-> mapping
lookups happen. Both are set in the ``feature_codes`` section of the config file.

Terminology
-----------
input / source codes
    Classification codes found in the source LAS/LAZ files.
application codes
    Consolidated codes the application reads and writes after mapping.
mapping
    ``{input_code: application_code}`` (see :attr:`FeatureCodes.mapping`).

Construction
------------
Built once at start-up from the config's ``feature_codes`` section (see
``params.FeatureCodesParam``) and handed to the pipeline in ``params.RunSettings``. The
example below uses made-up codes::

    >>> fc = FeatureCodes(
    ...     {187: 'Mainline Conductor', 210: 'Earth Wire', 215: 'Mainline Structure', 2: 'Ground'},
    ...     {
    ...         'wire': {'code': 15, 'input_feature_codes': [187, 210]},
    ...         'structure': {'code': 16, 'input_feature_codes': [215]},
    ...         'ground': {'code': 2, 'input_feature_codes': [2]},
    ...     },
    ...     required_application_feature_names=['wire', 'structure'],
    ... )
    >>> fc.input_codes
    [187, 210, 215, 2]
    >>> fc.application_codes
    [15, 16, 2]
    >>> fc.mapping
    {187: 15, 210: 15, 215: 16, 2: 2}

Pass ``None`` as the mapping to use the codes unchanged (no input -> application
remapping); features are then selected by their input name.

Selecting features
------------------
``fc[name]`` / ``fc[[names]]`` returns a new, narrowed ``FeatureCodes`` (names are
application names when a mapping exists). ``.code`` / ``.name`` are shortcuts valid
only when exactly one code is selected::

    >>> fc['wire'].input_codes
    [187, 210]
    >>> fc['wire'].application_codes
    [15]
    >>> fc['structure'].code
    215
    >>> fc['wire'].code
    Traceback (most recent call last):
    ValueError: Multiple feature codes found: [187, 210]. Use .input_codes or .application_codes ...

Shortcuts for the common "source codes in, application codes out" case::

    >>> fc.get_input_codes('wire')
    [187, 210]
    >>> fc.get_application_codes('wire')
    [15]

Read modes
----------
Calling the object, ``fc('source')`` or ``fc('application')``, returns a view for
the data being read. Use ``'source'`` for raw tiles (codes are still mapped on
write) and ``'application'`` for tiles already written with application codes
(no mapping, so ``.code`` is the application code)::

    >>> src = fc('source')[['structure', 'wire']]
    >>> src.input_codes, src.application_codes
    ([187, 210, 215], [15, 16])
    >>> app = fc('application')
    >>> app['wire'].code
    15
    >>> app.has_mapping
    False

Typical use in the pipeline
---------------------------
* Source classes that are reset and re-labelled::

      feature_codes.mapped_input_feature_codes

* The output code of a feature::

      feature_codes('application')['ground'].code

Validation
----------
Construction raises ``ValueError`` for codes outside [0, 255], duplicate codes or
names, an input code mapped to more than one application feature, or missing
``required_application_feature_names``. Mapped input codes that are absent from the
input features only log a warning. ``add_feature`` extends an instance in place.
"""

import logging
import typing as t

import numpy as np


logger = logging.getLogger(__name__)


class FeatureCodes:

    def __init__(
            self, 
            features: dict[int, str], 
            feature_mapping: dict[str, dict[str, t.Any]] | None,
            required_application_feature_names: list[str] | None = None,
        ):

        self._input_features = features
        self._feature_mapping = feature_mapping
        self._validate_args(required_application_feature_names)
    
    def __call__( # generate a new feature codes object for the specified read/write mode
            self, 
            read: t.Literal['source', 'application'], 
            *, 
            write: t.Optional[t.Literal['source', 'application']] = 'application',
        ):

        if read not in ['source', 'application']:
            raise ValueError(f'Invalid read mode: {read}. Must be "source" or "application".')
        if read == 'application':
            features = self.application_features
            if write == 'source':
               raise ValueError('Cannot read application feature codes and write source feature codes.')
            else:
                feature_mapping = None
        else:  # read == 'source'
            features = self.input_features
            if write == 'source':
                feature_mapping = None
            else:
                feature_mapping = self._feature_mapping


        return FeatureCodes(features=features, feature_mapping=feature_mapping)


    def __getitem__(self, item: str | list[str]):
        
        if isinstance(item, str):
            item = [item]
        if self.has_mapping:
            feature_mapping = {
                k: v for k, v in self._feature_mapping.items() 
                if k in item
            } 
            mapped_input_features = self.get_mapped_input_feature_codes(feature_mapping)
            features = {
                k: v for k, v in self.input_features.items() 
                if k in mapped_input_features
            } 
        else:
            features = {
                k: v for k, v in self.input_features.items() 
                if v in item
            } 
            feature_mapping = None

        if not features:
            raise ValueError(f'No features found for {item}.')

        return FeatureCodes(features=features, feature_mapping=feature_mapping)
    
    @property
    def code(self):
        if len(self.input_codes) == 1:
            return self.input_codes[0]
        else:
            raise ValueError(f'Multiple feature codes found: {self.input_codes}. Use .input_codes or .application_codes to access all codes.')
        
    @property
    def name(self):
        if len(self.input_names) == 1:
            return self.input_names[0]
        else:
            raise ValueError(f'Multiple feature names found: {self.input_names}. Use .input_names or .application_names to access all names.')
    
    @property
    def input_features(self):
        return self._input_features

    @property
    def input_codes(self):
        return list(self._input_features.keys())
    
    @property
    def input_names(self):
        return list(self.input_features.values())
    
    @property 
    def application_features(self):
        if self._feature_mapping is None:
            return self.input_features
        else:
            return {
                v['code']: k
                for k, v in self._feature_mapping.items() 
            }
    
    @property
    def application_codes(self):
        if self._feature_mapping is None:
            return self.input_codes
        return [v['code'] for v in self._feature_mapping.values()]
    
    @property
    def application_names(self):
        if self._feature_mapping is None:
            return self.input_names
        return list(self._feature_mapping.keys())
    
    @property
    def mapped_input_feature_codes(self):
        # list of feature codes in input laz
        # mapped to target feature codes used in application
        # (i.e. the input feature codes that are included in the laz reader pipeline)
        return self.get_mapped_input_feature_codes(self._feature_mapping)
        
    @property
    def input_to_application_feature_code_map(self):
        if self._feature_mapping is not None:
            return {
                y: x['code']
                for x in self._feature_mapping.values() 
                for y in x['input_feature_codes']
            }
        else:
            return {}
        
    @property
    def excluded_input_class_list(self):
        return [
            x for x in self.mapped_input_feature_codes
            if x not in self.input_codes
        ]
    
    @property
    def has_mapping(self):
        return self._feature_mapping is not None
    
    @property
    def mapping(self):
        return self.input_to_application_feature_code_map
    
    def add_feature(self, input_code: int, input_name: str, application_code: int | None = None):
        if input_code in self.input_codes:
            raise ValueError(f'Input feature code {input_code} already exists.')
        if input_name in self.input_names:
            raise ValueError(f'Input feature name {input_name} already exists.')
        if application_code is not None and application_code in self.application_codes:
            raise ValueError(f'Application feature code {application_code} already exists.')

        self._input_features[input_code] = input_name
        if application_code is not None:
            if self._feature_mapping is None:
                self._feature_mapping = {}
            self._feature_mapping[input_name] = {
                'code': application_code,
                'input_feature_codes': [input_code],
            }
    
    def get_input_codes(self, features: list[str] | str):

        return self(read='source', write='application')[features].input_codes

    
    def get_application_codes(self, features: list[str] | str):

        return self(read='source', write='application')[features].application_codes
    
    def get_mapped_input_feature_codes(self, mapping: dict[str, dict[str, t.Any]] | None = None):
        # list of feature codes in input laz
        # mapped to target feature codes used in application
        # (i.e. the input feature codes that are included in the laz reader pipeline)
        if mapping is not None:
            return [
                x for v in mapping.values() 
                for x in v['input_feature_codes']
            ]
        else:
            return []

    def _validate_args(self, required_application_feature_names):

        codes = (*self.input_codes, *self.application_codes)
        if any(isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255 for code in codes):
            raise ValueError('Source and application classification codes must be integers in [0, 255].')

        self._validate_no_duplicates(self.input_codes, 'input feature code')
        self._validate_no_duplicates(self.input_names, 'input feature name')
        self._validate_no_duplicates(self.application_codes, 'application feature code')
        self._validate_no_duplicates(self.application_names, 'application feature name')
        self._validate_no_duplicates(self.mapped_input_feature_codes, 'mapping of input to application')
        
        if len(self.excluded_input_class_list) > 0:
            logger.warning(
                f'The following feature codes are mapped but not included in '
                f'the input feature codes: {self.excluded_input_class_list}'
            )
        if required_application_feature_names is not None:
            missing = set(required_application_feature_names) - set(self.application_names)
            if missing:
                raise ValueError(
                    f'Missing required application feature names: {missing}. '
                    f'Available application feature names: {self.application_names}.'
                )
            
    @staticmethod
    def _validate_no_duplicates(codes, label):
        if len(codes) != len(set(codes)):
            non_unique = [x for x in codes if codes.count(x) > 1]
            raise ValueError(
                f'Found duplicate {label}: {list(set(non_unique))}'
            )