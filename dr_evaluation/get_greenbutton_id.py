import dataclient
m = dataclient.MDALClient("corbusier.cs.berkeley.edu:8088")

TED_meters = set(['jesse-turner-center'])

def get_greenbutton_id(sitename, time_start='2018-12-01T00:00:00Z', time_stop = '2018-12-02T00:00:00Z'):

    if sitename in TED_meters:
        definition = """ SELECT ?TED ?meter_uuid FROM %s WHERE {
                    ?TED rdf:type brick:Building_Electric_Meter .
                    ?TED bf:uuid ?meter_uuid
             }; """ % sitename
    else:
        definition = """SELECT ?gbm ?uuid FROM %s WHERE {
                    ?gbm rdf:type brick:Green_Button_Meter .
                    ?gbm bf:uuid ?uuid
                }; """ % sitename
        
    request = {
        "Variables": {
            "id": {
                "Definition": definition
            }
        }
    }

    request['Composition'] = ['id']
    request['Aggregation'] = {'id': ['MAX']}
    request['Time'] = {
        'Start': time_start,
        'End': time_stop,
        "Window": '24hr',
        "Aligned": True
    }
    resp = m.query(request)
    return resp.uuids

# print(get_greenbutton_id('ciee', "2018-01-01T10:00:00-07:00", "2018-08-12T10:00:00-07:00"))