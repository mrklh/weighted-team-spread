class Sqls:
    GET_GAME_DATA = """
    SELECT r.team_id, half*10000 + minute*100 + second, half, minute, 
           second, d.X_POS, d.Y_POS, d.JERSEY_NUMBER, d.hasball_team_id, p.name
        FROM tf_match_persecdata d, td_player p, tf_roster r, td_team t1, td_team t2, tf_match m
        WHERE d.team_id != 0 AND
              d.match_id = m.id AND 
              p.ID = r.PLAYER_ID AND
              d.JERSEY_NUMBER = r.JERSEY_NUMBER AND
              m.home_id = t1.id AND
              m.away_id = t2.id AND
              t1.name = %s AND
              t2.name = %s AND
              d.team_id = r.team_id AND
              d.match_id = r.match_id AND
              r.starting_status = 1 AND
              (d.X_POS != 0 AND d.Y_POS != 0)
        ORDER BY half, minute, second, r.team_id, name
    """

    GET_BALL_DATA = """
    SELECT m.id, half*10000 + minute*60 + second, half, minute, 
           second, d.X_POS, d.Y_POS, d.hasball_team_id, d.hasball_jersey_number
        FROM tf_match_persecdata d, td_team t1, td_team t2, tf_match m
        WHERE d.team_id = 0 AND
              d.match_id = m.id AND 
              m.home_id = t1.id AND
              m.away_id = t2.id AND
              t1.name = %s AND
              t2.name = %s AND
              (d.X_POS != 0 AND d.Y_POS != 0) AND
              d.hasball_team_id != 0
        ORDER BY half, minute, second
    """

    GET_TEAM_DATA = """
        SELECT id FROM td_team WHERE name = '%s'
    """

    GET_PLYR_DATA = """
        select r.team_id, r.jersey_number, p.name from tf_roster r, tf_match m, td_player p 
        where r.MATCH_ID = m.id and m.id = %s and r.PLAYER_ID = p.id
        and r.STARTING_STATUS = 1
    """

    GET_GAMES_DATA = """
        SELECT p.MATCH_ID, t1.NAME, t1.ID, t2.NAME, t2.ID, m.HOME_SCORE, m.AWAY_SCORE 
            FROM tf_match_persecdata p, tf_match m, td_team t1, td_team t2
        
        WHERE p.MATCH_ID = m.ID AND
              t2.ID = m.AWAY_ID AND
              t1.ID = m.HOME_ID AND
              t1.ID = 3 AND
              t2.ID = 86
        GROUP BY(p.MATCH_ID) LIMIT 10
    """

    GET_TEAM_LENGTH_DATA = """
    SELECT sum(diff)/5400 as avg_diff, TEAM_ID from (
        select 
        HALF*10000 + MINUTE*100 + SECOND, TEAM_ID, min(X_POS), max(X_POS),
        max(X_POS) - min(X_POS) as diff
        from tf_match_persecdata 
        WHERE MATCH_ID = %s and X_POS != 0 and TEAM_ID != 0
        group by HALF*10000 + MINUTE*100 + SECOND, TEAM_ID) as foo group by TEAM_ID
    """

    GET_EVENT_DATA = """
    select e.HALF, e.MINUTE, e.SECOND, e.JERSEY_IN, e.JERSEY_OUT, d.NAME 
    from tf_sync_event e, td_sync_dictionary d 
    where e.MATCH_ID=%s and e.TYPE = d.ID
    """
