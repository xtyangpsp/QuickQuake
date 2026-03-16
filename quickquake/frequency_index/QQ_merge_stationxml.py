from pathlib import Path
from copy import deepcopy
from obspy import read_inventory, Inventory
from obspy.core.inventory import Network, Station


DATA_DIR = Path("/home/elizabeth/soft/src/QuickQuake/data")
OUTPUT_XML = DATA_DIR / "merged" / "input" / "stations_merged.xml"


def is_chunk_folder(path: Path) -> bool:
    return path.is_dir() and path.name != "merged"


def collect_stationxml_files(data_dir: Path):
    xml_files = []
    for sub in sorted(data_dir.iterdir()):
        if not is_chunk_folder(sub):
            continue
        xml_path = sub / "stations.xml"
        if xml_path.exists():
            xml_files.append(xml_path)
    return xml_files


def channel_key(ch):
    return (
        ch.location_code,
        ch.code,
        str(ch.start_date) if ch.start_date else None,
        str(ch.end_date) if ch.end_date else None,
        round(ch.latitude, 6) if ch.latitude is not None else None,
        round(ch.longitude, 6) if ch.longitude is not None else None,
        round(ch.elevation, 3) if ch.elevation is not None else None,
        round(ch.depth, 3) if ch.depth is not None else None,
        round(ch.azimuth, 3) if ch.azimuth is not None else None,
        round(ch.dip, 3) if ch.dip is not None else None,
        getattr(ch.response, "instrument_sensitivity", None).value
        if ch.response and ch.response.instrument_sensitivity
        else None,
    )


def station_key(sta):
    return (
        sta.code,
        round(sta.latitude, 6) if sta.latitude is not None else None,
        round(sta.longitude, 6) if sta.longitude is not None else None,
        round(sta.elevation, 3) if sta.elevation is not None else None,
        str(sta.start_date) if sta.start_date else None,
        str(sta.end_date) if sta.end_date else None,
    )


def merge_inventories_dedup(xml_files):
    network_map = {}

    for xml_file in xml_files:
        print(f"Reading: {xml_file}")
        inv = read_inventory(str(xml_file))

        for net in inv.networks:
            if net.code not in network_map:
                new_net = Network(
                    code=net.code,
                    stations=[],
                    description=net.description,
                    start_date=net.start_date,
                    end_date=net.end_date,
                    total_number_of_stations=net.total_number_of_stations,
                    selected_number_of_stations=net.selected_number_of_stations,
                )
                network_map[net.code] = {
                    "network": new_net,
                    "stations": {}
                }

            net_entry = network_map[net.code]

            for sta in net.stations:
                s_key = station_key(sta)

                if s_key not in net_entry["stations"]:
                    new_sta = Station(
                        code=sta.code,
                        latitude=sta.latitude,
                        longitude=sta.longitude,
                        elevation=sta.elevation,
                        creation_date=sta.creation_date,
                        site=deepcopy(sta.site),
                        vault=sta.vault,
                        geology=sta.geology,
                        equipments=deepcopy(sta.equipments),
                        operators=deepcopy(sta.operators),
                        water_level=sta.water_level,
                        description=sta.description,
                        comments=deepcopy(sta.comments),
                        start_date=sta.start_date,
                        end_date=sta.end_date,
                        restricted_status=sta.restricted_status,
                        alternate_code=sta.alternate_code,
                        historical_code=sta.historical_code,
                        data_availability=deepcopy(sta.data_availability),
                        identifiers=deepcopy(sta.identifiers),
                        source_id=sta.source_id,
                        channels=[],
                    )
                    net_entry["stations"][s_key] = {
                        "station": new_sta,
                        "channels": set()
                    }

                sta_entry = net_entry["stations"][s_key]

                for ch in sta.channels:
                    c_key = channel_key(ch)
                    if c_key not in sta_entry["channels"]:
                        sta_entry["station"].channels.append(deepcopy(ch))
                        sta_entry["channels"].add(c_key)

    merged_networks = []
    for net_code, net_entry in network_map.items():
        net_obj = net_entry["network"]
        net_obj.stations = [v["station"] for v in net_entry["stations"].values()]
        merged_networks.append(net_obj)

    merged_inv = Inventory(networks=merged_networks, source="QuickQuake merged StationXML")
    return merged_inv


def summarize_inventory(inv):
    n_networks = len(inv.networks)
    n_stations = sum(len(net.stations) for net in inv.networks)
    n_channels = sum(len(sta.channels) for net in inv.networks for sta in net.stations)

    print("\nMerged inventory summary")
    print(f"Networks : {n_networks}")
    print(f"Stations : {n_stations}")
    print(f"Channels : {n_channels}")


def main():
    OUTPUT_XML.parent.mkdir(parents=True, exist_ok=True)

    xml_files = collect_stationxml_files(DATA_DIR)
    print(f"Found {len(xml_files)} StationXML files")

    if not xml_files:
        raise FileNotFoundError(f"No stations.xml files found in {DATA_DIR}")

    merged_inv = merge_inventories_dedup(xml_files)
    summarize_inventory(merged_inv)

    print(f"\nWriting merged StationXML to: {OUTPUT_XML}")
    merged_inv.write(str(OUTPUT_XML), format="STATIONXML")
    print("Done.")


if __name__ == "__main__":
    main()