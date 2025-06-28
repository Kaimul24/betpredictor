# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class BatterStat(scrapy.Item):
    player_id = scrapy.Field()
    name = scrapy.Field()
    team = scrapy.Field()
    games = scrapy.Field()
    ab = scrapy.Field()
    pa = scrapy.Field()
    ops = scrapy.Field()
    bb_k = scrapy.Field()
    wrc_plus = scrapy.Field()
    barrel_percent = scrapy.Field()
    hard_hit = scrapy.Field()
    war = scrapy.Field()
    baserunning = scrapy.Field()
    scraped_at = scrapy.Field()
    date = scrapy.Field()


class PitcherStat(scrapy.Item):
    player_id = scrapy.Field()
    name = scrapy.Field()
    team = scrapy.Field()
    games = scrapy.Field()
    era = scrapy.Field()
    ip = scrapy.Field()
    k_percent = scrapy.Field()
    bb_percent = scrapy.Field()
    barrel_percent = scrapy.Field()
    hard_hit = scrapy.Field()
    war = scrapy.Field()
    siera = scrapy.Field()
    fip = scrapy.Field()
    scraped_at = scrapy.Field()
    date = scrapy.Field()

class OddsItem(scrapy.Item):
    date = scrapy.Field()
    away_team = scrapy.Field()
    home_team = scrapy.Field()
    away_starter = scrapy.Field()
    home_starter = scrapy.Field()
    away_score = scrapy.Field()
    home_score = scrapy.Field()
    winner = scrapy.Field()
    sportsbook = scrapy.Field()
    away_odds = scrapy.Field()
    home_odds = scrapy.Field()
