from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import next
from builtins import int

import os
import unittest
import datetime

from traildb import TrailDB, TrailDBConstructor, tdb_item_field, tdb_item_val
from traildb import TrailDBError, TrailDBCursor


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1', 'field2'])
        cons.add(self.uuid, 1, ['a', '1'])
        cons.add(self.uuid, 2, ['b', '2'])
        cons.add(self.uuid, 3, ['c', '3'])
        cons.finalize()

    def tearDown(self):
        os.unlink('testtrail.tdb')

    def test_trails(self):
        db = TrailDB('testtrail')
        self.assertEqual(1, db.num_trails)

        trail = db.trail(0)
        self.assertIsInstance(trail, TrailDBCursor)
        events = list(trail)  # Force evaluation of generator
        self.assertEqual(3, len(events))
        
        n = 0

        for event in events:
            self.assertTrue(hasattr(event, 'time'))
            self.assertTrue(hasattr(event, 'field1'))
            self.assertTrue(hasattr(event, 'field2'))
            self.assertEqual(str(n+1), event.field2)

            with self.assertRaises(AttributeError):
                event.missing_field

            n += 1

        self.assertEqual(3, n)

    def test_trails_selected_uuids(self):
        uuids = ["02345678123456781234567812345678",
                 "12345678123456781234567812345678",
                 "22345678123456781234567812345678",
                 "32345678123456781234567812345678",
                 "42345678123456781234567812345678"]
        cons = TrailDBConstructor('whitelist_testtrail', ['field1', 'field2'])
        for uuid in uuids:
            cons.add(uuid, 1, ['a', '1'])
            cons.add(uuid, 2, ['b', '2'])
            cons.add(uuid, 3, ['c', '3'])
        cons.finalize()
        
        tdb = TrailDB('whitelist_testtrail')
        whitelist = [uuids[0],
                     uuids[3],
                     uuids[4]]
        
        expected_length = 3
        for trail_uuid, trail_events in tdb.trails(selected_uuids=whitelist):
            trail_events = list(trail_events)
            self.assertEqual(len(trail_events),
                             expected_length)

    def test_crumbs(self):
        db = TrailDB('testtrail.tdb')

        n = 0
        for uuid, trail in db.trails():
            n += 1
            self.assertEqual(self.uuid, uuid)
            self.assertIsInstance(trail, TrailDBCursor)
            self.assertEqual(3, len(list(trail)))

        self.assertEqual(1, n)

    def test_silly_open(self):
        self.assertTrue(os.path.exists('testtrail.tdb'))
        self.assertFalse(os.path.exists('testtrail'))

        db1 = TrailDB('testtrail.tdb')
        db2 = TrailDB('testtrail')

        with self.assertRaises(TrailDBError):
            TrailDB('foo.tdb')

    def test_fields(self):
        db = TrailDB('testtrail')
        self.assertEqual(['time', 'field1', 'field2'], db.fields)

    def test_uuids(self):
        db = TrailDB('testtrail')
        self.assertEqual(0, db.get_trail_id(self.uuid))
        self.assertEqual(self.uuid, db.get_uuid(0))
        self.assertTrue(self.uuid in db)

    def test_lexicons(self):
        db = TrailDB('testtrail')

        # First field
        self.assertEqual(4, db.lexicon_size(1))
        self.assertEqual(['a', 'b', 'c'], list(db.lexicon(1)))

        # Second field
        self.assertEqual(['1', '2', '3'], list(db.lexicon(2)))

        with self.assertRaises(TrailDBError):
            db.lexicon(3)  # Out of bounds

    def test_metadata(self):
        db = TrailDB('testtrail.tdb')
        self.assertEqual(1, db.min_timestamp())
        self.assertEqual(3, db.max_timestamp())
        self.assertEqual((1, 3), db.time_range())

        self.assertEqual((1, 3), db.time_range(parsetime=False))


    def test_apply_whitelist(self):
        uuids = ["02345678123456781234567812345678",
                 "12345678123456781234567812345678",
                 "22345678123456781234567812345678",
                 "32345678123456781234567812345678",
                 "42345678123456781234567812345678"]
        cons = TrailDBConstructor('whitelist_testtrail', ['field1', 'field2'])
        for uuid in uuids:
            cons.add(uuid, 1, ['a', '1'])
            cons.add(uuid, 2, ['b', '2'])
            cons.add(uuid, 3, ['c', '3'])
        cons.finalize()
        
        tdb = TrailDB('whitelist_testtrail')
        whitelist = [uuids[0],
                     uuids[3],
                     uuids[4]]
        tdb.apply_whitelist(whitelist)
        found_trails = list(tdb.trails(parsetime=False))

        self.assertEqual(len(found_trails), len(uuids))
        for trail_uuid, trail_events in found_trails:
            if trail_uuid in whitelist:
                expected_length = 3
            else:
                expected_length = 0
                
            trail_events = list(trail_events)
            self.assertEqual(len(trail_events),
                             expected_length)

    def test_apply_blacklist(self):
        uuids = ["02345678123456781234567812345678",
                 "12345678123456781234567812345678",
                 "22345678123456781234567812345678",
                 "32345678123456781234567812345678",
                 "42345678123456781234567812345678"]
        cons = TrailDBConstructor('blacklist_testtrail', ['field1', 'field2'])
        for uuid in uuids:
            cons.add(uuid, 1, ['a', '1'])
            cons.add(uuid, 2, ['b', '2'])
            cons.add(uuid, 3, ['c', '3'])
        cons.finalize()
        
        tdb = TrailDB('blacklist_testtrail')
        blacklist = [uuids[1],
                     uuids[2]]
        tdb.apply_blacklist(blacklist)
        found_trails = list(tdb.trails(parsetime=False))

        for trail_uuid, trail_events in found_trails:
            if trail_uuid in blacklist:
                expected_length = 0
            else:
                expected_length = 3
                
            trail_events = list(trail_events)
            self.assertEqual(len(trail_events),
                             expected_length)

    def test_apply_whitelist(self):
        uuids = ["02345678123456781234567812345678",
                 "12345678123456781234567812345678",
                 "22345678123456781234567812345678",
                 "32345678123456781234567812345678",
                 "42345678123456781234567812345678"]
        cons = TrailDBConstructor('whitelist_testtrail', ['field1', 'field2'])
        for uuid in uuids:
            cons.add(uuid, 1, ['a', '1'])
            cons.add(uuid, 2, ['b', '2'])
            cons.add(uuid, 3, ['c', '3'])
        cons.finalize()
        
        tdb = TrailDB('whitelist_testtrail')
        whitelist = [uuids[0],
                     uuids[3],
                     uuids[4]]
        tdb.apply_whitelist(whitelist)
        found_trails = list(tdb.trails(parsetime=False, distinct_cursors=True))

        self.assertEqual(len(found_trails), len(uuids))
        for trail_uuid, trail_events in found_trails:
            if trail_uuid in whitelist:
                expected_length = 3
            else:
                expected_length = 0
                
            trail_events = list(trail_events)
            self.assertEqual(len(trail_events),
                             expected_length)

    def test_apply_blacklist(self):
        uuids = ["02345678123456781234567812345678",
                 "12345678123456781234567812345678",
                 "22345678123456781234567812345678",
                 "32345678123456781234567812345678",
                 "42345678123456781234567812345678"]
        cons = TrailDBConstructor('blacklist_testtrail', ['field1', 'field2'])
        for uuid in uuids:
            cons.add(uuid, 1, ['a', '1'])
            cons.add(uuid, 2, ['b', '2'])
            cons.add(uuid, 3, ['c', '3'])
        cons.finalize()
        
        tdb = TrailDB('blacklist_testtrail')
        blacklist = [uuids[1],
                     uuids[2]]
        tdb.apply_blacklist(blacklist)
        found_trails = list(tdb.trails(parsetime=False, distinct_cursors=True))

        for trail_uuid, trail_events in found_trails:
            if trail_uuid in blacklist:
                expected_length = 0
            else:
                expected_length = 3
                
            trail_events = list(trail_events)
            self.assertEqual(len(trail_events),
                             expected_length)

class TestFilter(unittest.TestCase):

    def setUp(self):
        uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1', 'field2', 'field3'])
        cons.add(uuid, 1, ['a', '1', 'x'])
        cons.add(uuid, 2, ['b', '2', 'x'])
        cons.add(uuid, 3, ['c', '3', 'y'])
        cons.add(uuid, 4, ['d', '4', 'x'])
        cons.add(uuid, 5, ['e', '5', 'x'])
        tdb = cons.finalize()

    def tearDown(self):
        os.unlink('testtrail.tdb')

    def test_simple_disjunction(self):
        tdb = TrailDB('testtrail')
        # test shorthand notation (not a list of lists)
        events = list(
            tdb.trail(0, event_filter=[('field1', 'a'), ('field2', '4')]))
        self.assertEqual(len(events), 2)
        self.assertEqual((events[0].field1, events[0].field2), ('a', '1'))
        self.assertEqual((events[1].field1, events[1].field2), ('d', '4'))

    def test_negation(self):
        tdb = TrailDB('testtrail')
        events = list(tdb.trail(0, event_filter=[('field3', 'x', True)]))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].field1, events[0].field2,
                          events[0].field3), ('c', '3', 'y'))

    def test_conjunction(self):
        tdb = TrailDB('testtrail')
        events = list(
            tdb.trail(0, event_filter=[[('field1', 'e'), ('field1', 'c')],
                                       [('field3', 'y', True)]]))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].field1, events[0].field2), ('e', '5'))

    def test_time_range(self):
        tdb = TrailDB('testtrail')
        events = list(tdb.trail(0,
                                event_filter=[[(2, 4)]],
                                parsetime=False))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].time, 2)
        self.assertEqual(events[1].time, 3)

    def test_filter_object(self):
        tdb = TrailDB('testtrail')
        obj = tdb.create_filter([[('field1', 'e'), ('field1', 'c')],
                                 [('field3', 'y', True)]])
        events = list(tdb.trail(0, event_filter=obj))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].field1, events[0].field2), ('e', '5'))
        events = list(tdb.trail(0, event_filter=obj))
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].field1, events[0].field2), ('e', '5'))


class TestCons(unittest.TestCase):
    def test_cursor(self):
        uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1', 'field2'])
        cons.add(uuid, 1, ['a', '1'])
        cons.add(uuid, 2, ['b', '2'])
        cons.add(uuid, 3, ['c', '3'])
        cons.add(uuid, 4, ['d', '4'])
        cons.add(uuid, 5, ['e', '5'])
        tdb = cons.finalize()

        with self.assertRaises(IndexError):
            tdb.get_trail_id('12345678123456781234567812345679')

        trail = tdb.trail(tdb.get_trail_id(uuid))
        with self.assertRaises(TypeError):
            len(trail)

        j = 1
        for event in trail:
            self.assertEqual(j, int(event.field2))
            self.assertEqual(j, int(event.time))
            j += 1
        self.assertEqual(6, j)

        # Iterator is empty now
        self.assertEqual([], list(trail))

        field1_values = [e.field1 for e in tdb.trail(tdb.get_trail_id(uuid))]
        self.assertEqual(['a', 'b', 'c', 'd', 'e'], field1_values)

    def test_cursor_parsetime(self):
        uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1'])

        events = [(datetime.datetime(2016, 1, 1, 1, 1), ['1']),
                  (datetime.datetime(2016, 1, 1, 1, 2), ['2']),
                  (datetime.datetime(2016, 1, 1, 1, 3), ['3'])]
        [cons.add(uuid, time, fields) for time, fields in events]
        tdb = cons.finalize()

        timestamps = [e.time for e in tdb.trail(0, parsetime=True)]

        self.assertIsInstance(timestamps[0], datetime.datetime)
        self.assertEqual([time for time, _ in events], timestamps)
        self.assertEqual(tdb.time_range(True), (events[0][0], events[-1][0]))

    def test_binarydata(self):
        binary = b'\x00\x01\x02\x00\xff\x00\xff'
        uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1'])
        cons.add(uuid, 123, [binary])
        tdb = cons.finalize(decode=False)
        self.assertEqual(list(tdb[0])[0].field1, binary)

    def test_cons(self):
        uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1', 'field2'])
        cons.add(uuid, 123, ['a'])
        cons.add(uuid, 124, ['b', 'c'])
        tdb = cons.finalize()

        self.assertEqual(0, tdb.get_trail_id(uuid))
        self.assertEqual(uuid, tdb.get_uuid(0))
        self.assertEqual(1, tdb.num_trails)
        self.assertEqual(2, tdb.num_events)
        self.assertEqual(3, tdb.num_fields)

        crumbs = list(tdb.trails())
        self.assertEqual(1, len(crumbs))
        self.assertEqual(uuid, crumbs[0][0])
        self.assertTrue(tdb[uuid])
        self.assertTrue(uuid in tdb)
        self.assertFalse('00000000000000000000000000000000' in tdb)
        with self.assertRaises(IndexError):
            tdb['00000000000000000000000000000000']

        trail = list(crumbs[0][1])

        self.assertEqual(123, trail[0].time)
        self.assertEqual('a', trail[0].field1)
        self.assertEqual('', trail[0].field2)  # TODO: Should this be None?

        self.assertEqual(124, trail[1].time)
        self.assertEqual('b', trail[1].field1)
        self.assertEqual('c', trail[1].field2)

    def test_items(self):
        uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1', 'field2'])
        cons.add(uuid, 123, ['a', 'x' * 2048])
        cons.add(uuid, 124, ['b', 'y' * 2048])
        tdb = cons.finalize()

        cursor = tdb.trail(0, rawitems=True)
        event = next(cursor)
        self.assertEqual(tdb.get_item_value(event.field1), 'a')
        self.assertEqual(tdb.get_item_value(event.field2), 'x' * 2048)
        self.assertEqual(tdb.get_item('field1', 'a'), event.field1)
        self.assertEqual(tdb.get_item('field2', 'x' * 2048), event.field2)
        event = next(cursor)
        self.assertEqual(tdb.get_item_value(event.field1), 'b')
        self.assertEqual(tdb.get_item_value(event.field2), 'y' * 2048)
        self.assertEqual(tdb.get_item('field1', 'b'), event.field1)
        self.assertEqual(tdb.get_item('field2', 'y' * 2048), event.field2)

        cursor = tdb.trail(0, rawitems=True)
        event = next(cursor)
        field = tdb_item_field(event.field1)
        val = tdb_item_val(event.field1)
        self.assertEqual(tdb.get_value(field, val), 'a')
        field = tdb_item_field(event.field2)
        val = tdb_item_val(event.field2)
        self.assertEqual(tdb.get_value(field, val), 'x' * 2048)
        event = next(cursor)
        field = tdb_item_field(event.field1)
        val = tdb_item_val(event.field1)
        self.assertEqual(tdb.get_value(field, val), 'b')
        field = tdb_item_field(event.field2)
        val = tdb_item_val(event.field2)
        self.assertEqual(tdb.get_value(field, val), 'y' * 2048)

    def test_append(self):
        uuid = '12345678123456781234567812345678'
        cons = TrailDBConstructor('testtrail', ['field1'])
        cons.add(uuid, 123, ['foobarbaz'])
        tdb = cons.finalize()

        cons = TrailDBConstructor('testtrail2', ['field1'])
        cons.add(uuid, 124, ['barquuxmoo'])
        cons.append(tdb)
        tdb = cons.finalize()

        self.assertEqual(2, tdb.num_events)
        uuid, trail = list(tdb.trails())[0]
        trail = list(trail)
        self.assertEqual([123, 124], [e.time for e in trail])
        self.assertEqual(['foobarbaz', 'barquuxmoo'],
                         [e.field1 for e in trail])

    def tearDown(self):
        try:
            os.unlink('testtrail.tdb')
            os.unlink('testtrail2.tdb')
        except:
            pass


if __name__ == '__main__':
    unittest.main()
