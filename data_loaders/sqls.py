# -*- coding: utf-8 -*-

class Sqls:
    GET_GAME_DATA = """
    SELECT r.team_id, half*10000 + minute*100 + second, half, minute, 
           second, d.X_POS, d.Y_POS, d.JERSEY_NUMBER, d.hasball_team_id, r.starting_status, d.SPEED, p.name
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
              (d.X_POS != 0 AND d.Y_POS != 0) AND 
              d.X_POS IS NOT NULL AND d.Y_POS IS NOT NULL
              AND d.team_id IS NOT NULL 
              AND m.SEASON_ID = 11075
        ORDER BY half, minute, second, r.team_id, name
    """

    GET_BALL_DATA = """
    SELECT m.id, half*10000 + minute*100 + second, half, minute, 
           second, d.X_POS, d.Y_POS, d.hasball_team_id, d.hasball_jersey_number
        FROM tf_match_persecdata d, td_team t1, td_team t2, tf_match m
        WHERE d.team_id = 0 AND
              d.match_id = m.id AND 
              m.home_id = t1.id AND
              m.away_id = t2.id AND
              t1.name = %s AND
              t2.name = %s AND
              (d.X_POS != 0 AND d.Y_POS != 0) AND
              (d.X_POS IS NOT NULL AND d.Y_POS IS NOT NULL) AND
              d.hasball_team_id != 0
              AND m.SEASON_ID = 11075
        ORDER BY half, minute, second
    """

    GET_BALL_POS_DATA = """
    SELECT half*10000 + minute*100 + second, half, minute, 
       second, d.X_POS, d.Y_POS, d.hasball_team_id
    FROM tf_match_persecdata d, tf_match m
    WHERE d.team_id = 0 AND
          d.match_id = m.id AND 
          d.match_id = %d AND
          X_POS IS NOT NULL AND Y_POS IS NOT NULL AND
          (d.X_POS != 0 AND d.Y_POS != 0)
          AND m.SEASON_ID = 11075
    ORDER BY half, minute, second
    """

    GET_TEAM_DATA = """
        SELECT id FROM td_team WHERE name = '%s'
    """

    GET_PLYR_DATA = """
        select r.team_id, r.jersey_number, p.name from tf_roster r, tf_match m, td_player p 
        where r.MATCH_ID = m.id and m.id = %s and r.PLAYER_ID = p.id
        AND m.SEASON_ID = 11075
        and r.STARTING_STATUS = 1
    """

    GET_FIRST_ELEVEN = """
    SELECT r.JERSEY_NUMBER, p.NAME FROM tr_18.tf_roster r, tr_18.td_player p 
    WHERE r.PLAYER_ID = p.ID AND r.MATCH_ID = %s AND r.STARTING_STATUS = 1
    AND TEAM_ID = 3 ORDER BY r.POSITION_ID, JERSEY_NUMBER
    """

    GET_GAMES_DATA = """
        SELECT p.MATCH_ID, t1.NAME, t1.ID, t2.NAME, t2.ID, m.HOME_SCORE, m.AWAY_SCORE 
            FROM tf_match_persecdata p, tf_match m, td_team t1, td_team t2
        
        WHERE p.MATCH_ID = m.ID AND
              t2.ID = m.AWAY_ID AND
              t1.ID = m.HOME_ID AND
              (HOME_ID = 3 AND AWAY_ID = 105)
              AND m.SEASON_ID = 11075
        GROUP BY(p.MATCH_ID) ORDER BY m.MATCH_DATE DESC
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
    SELECT d.half*10000 + d.minute*100 + d.second, d.TEAM_ID, d.JERSEY_NUMBER, e.ID 
    FROM tf_tagger_data d, td_tagger_event e
    WHERE 
    d.MATCH_ID = %d AND 
    e.ID = d.EVENT_ID AND
    (e.ID = 10 or e.ID = 70 or e.ID = 72 or e.ID = 12)
    """ # TODO using player id instead of jersey number

    GET_SUBS_DATA = """
    SELECT e.team_id, e.JERSEY_OUT as jout, r1.PLAYER_ID as idout, p1.NAME as nout, 
        e.JERSEY_IN as jin, r2.PLAYER_ID as idin, p2.NAME as nin
    FROM tf_sync_event e, tf_roster r1, tf_roster r2, td_player p1, td_player p2
    WHERE e.match_id = %d and jersey_in != -1 and
    e.JERSEY_OUT = r1.jersey_number and e.jersey_in = r2.jersey_number
    and r1.match_id = e.match_id and r2.match_id = e.match_id
    and r1.team_id = e.team_id and r2.team_id = e.team_id and
    p1.id = r1.player_id and p2.id = r2.player_id
    """

    GET_SPRINT_DATA = """
    select e.* from curr_hirsprint_data_exp e, tf_match m where e.MATCH_ID = m.ID and m.ID = %d and IS_SPRINT=1
    """
