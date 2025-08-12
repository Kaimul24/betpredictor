#!/usr/bin/env python3

import statsapi
import json
from pprint import pprint

# Test a single game to see the structure
params = {
    'sportId': 1,
    'startDate': '04/01/2021',
    'endDate': '04/01/2021',
    'hydrate': 'team,weather,statusFlags,venue(timezone,location,elevation),probablePitcher'
}

data = statsapi.get('schedule', params=params, force=False)
dates = data['dates']

if dates:
    games = dates[0]['games']
    if games:
        game = games[0]
        print("Sample game structure:")
        print("=" * 50)
        print(f"Game ID: {game['gamePk']}")
        print(f"Date: {game['officialDate']}")
        
        print("\nAway team probable pitcher:")
        pprint(game['teams']['away']['probablePitcher'])
        
        print("\nHome team probable pitcher:")
        pprint(game['teams']['home']['probablePitcher'])
        
        print("\nFull away team data keys:")
        print(list(game['teams']['away'].keys()))
        
        print("\nFull home team data keys:")
        print(list(game['teams']['home'].keys()))
