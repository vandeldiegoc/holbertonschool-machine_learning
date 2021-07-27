#!/usr/bin/env python3
"""module"""
import requests


def sentientPlanets():
    """method that returns the list of names of
    the home planets of all sentient species."""
    r = 'https://swapi-api.hbtn.io/api/species'
    planet = []
    state = True
    while state:
        data = requests.get(r).json()
        for species in data['results']:
            if (species['designation'] == 'sentient' or
                species['classification'] == 'sentient') and\
                     species['homeworld'] is not None:
                hw = requests.get(species['homeworld']).json()
                planet.append(hw['name'])
        r = data['next']
        if r is None:
            state = False
    return planet
