'''CSC108 Assignment 3: Social Networks'''

from typing import List, Tuple, Dict, TextIO


def load_profiles(profiles_file: TextIO, person_to_friends: Dict[str, List[str]], \
                  person_to_networks: Dict[str, List[str]]) -> None:
    '''Update the person_to_friends dictionary and the person_to_networks
    dictionary to include data from profiles_file.
    Docstring examples not given since the result depends on input data.
    '''
    fileContent = profiles_file.readlines()
    numOfLines = len(fileContent)
    start = 0
    read_content = []
    for num in range(numOfLines):
        if fileContent[num].strip() is None:
            num_list = numOfLines[start:num]
            start = num + 1
            read_content.append(num_list)
    read_content.append(numOfLines[start:])
    for item in read_content:
        if len(item) > 1:
            key = str(item[0].strip().split(',')[1]) + " " +  str(item[0].strip().split(',')[0])
            for num in range(1, len(item)):
                if len(item[num].strip().split(',')) > 1:
                    name = str(item[num].strip().split(',')[1]) + " " + str(item[num].strip().split(',')[0])
                    if key in person_to_friends.keys():  # insure if key has then append  else create
                        person_to_friends[key].append(name)
                        person_to_friends[key].sort()  # sort by alpha
                    else:
                        person_to_friends[key] = [name]
                else:
                    if key in person_to_networks.keys():
                        person_to_networks[key].append(item[num])
                        person_to_friends[key].sort()  # sort by alpha
                    else:
                        person_to_networks[key] = [item[num]]





def get_average_friend_count(person_to_friends: Dict[str, List[str]]) -> float:
    '''
    '''
    num_person = len(person_to_friends.keys())
    num_friend = 0
    for item in person_to_friends.values():
        num_friend = num_friend + len(item)
    return float(num_friend/num_person)

    
def get_families(person_to_friends: Dict[str, List[str]]) -> Dict[str, List[str]]:
    '''
    '''
    all_first_name = []
    all_last_name = []
    for item in list(person_to_friends.keys()):
         all_first_name.append(item.split(" ")[0])
         all_last_name.append(item.split(" ")[1])
    for item in list(person_to_friends.values()):
        all_first_name.extend(item.split(" ")[0])
        all_last_name.extend(item.split(" ")[1])
    final_dict = {}
    for num in range(len(all_last_name)):
        if all_last_name[num] in final_dict.keys():
            final_dict[all_last_name[num]].append(all_first_name[num])
        else:
            final_dict[all_last_name[num]] = [all_first_name[num]]
    # delete the same name
    for item in final_dict.keys():
        final_dict[item] = list(set(list(final_dict[item]).sort()))

def invert_network(person_to_networks: Dict[str, List[str]]) -> Dict[str, List[str]]:
    '''
    '''
    new_dict = {}
    for item in person_to_networks.items():
        for factor in item[1]:
            if factor in new_dict.keys():
                new_dict[factor].append(item[0]).sort()
            else:
                new_dict[factor] = [item[0]]
    return new_dict


def get_friends_of_friends(person_to_friends: Dict[str, List[str]], \
                           person: str) -> List[str]:
    '''
    '''
    if person in person_to_friends.keys():
        return person_to_friends[person].sort()
    else:
        return None

def make_recommendations(person: str, person_to_friends: Dict[str, List[str]], \
                         person_to_networks: Dict[str, List[str]]) \
                         -> List[Tuple[str, int]]:
    '''
    '''

    
def is_network_connected(person_to_friends: Dict[str, List[str]]) -> bool:
    '''
    '''


if __name__ == '__main__':
    # Do not move this code out of the __main__ block, and do not add any other 
    # testing code outside of the __main__ block.
    import doctest
    doctest.testmod()
