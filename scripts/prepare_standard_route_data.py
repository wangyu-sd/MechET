"""Translate complete pinned source TRAIN/VALID without route curation or models."""
import argparse
import json
from pathlib import Path
from mechet.route_standard_data import prepare_standard_route_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','validation-source','output','stock-file','source-manifest','release-checksums'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--release',required=True)
    parser.add_argument('--stock-condition',required=True)
    parser.add_argument('--expected-stock-sha256',required=True)
    parser.add_argument('--benchmark-targets',nargs='*',type=Path,default=[])
    parser.add_argument('--coverage-stock',action='append',default=[],metavar='NAME=PATH')
    parser.add_argument('--source-audit',type=Path)
    args=vars(parser.parse_args())
    coverage={}
    for item in args.pop('coverage_stock'):
        name,separator,value=item.partition('=')
        if not separator or not name or not value:parser.error('--coverage-stock requires NAME=PATH')
        if name in coverage:parser.error('duplicate coverage stock '+name)
        coverage[name]=Path(value)
    report=prepare_standard_route_data(**args,coverage_stocks=coverage,
        progress=lambda event:print(json.dumps(event),flush=True))
    print(json.dumps(dict(event='standard-route-complete',source_records=report['source_records'],
        training_targets=report['training_targets'],validation_targets=report['validation_targets'],
        private_reaction_occurrences=report['private_reaction_occurrences'])),flush=True)


if __name__=='__main__':main()
