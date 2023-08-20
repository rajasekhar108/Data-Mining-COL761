#include<bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;
string enc_file,op_fi;
unordered_set<string>keys;
void givealllines(FILE *f,list<string>&alllines,int &c,int &no_fp)
{
    string line;

    while (true) {
        int ch = getc_unlocked(f);
        if (ch == EOF) {
            if (!line.empty()) {
                alllines.push_back(line);
            }
            break;
        } else if (ch == '\n') {
            c++;
            no_fp++;
            alllines.push_back(line);
            line.clear();
        } else {
            if(static_cast<char>(ch)==' ') c++;
            line += static_cast<char>(ch);
        }
    }
}
void seperate_map_and_data(list<string>&all_lines_in_file,list<string>&mapping_data,int &items_in_map,int &no_of_mappings)
{
    while(all_lines_in_file.back()!="1mapping_starts1")
    {
        mapping_data.push_back(all_lines_in_file.back());
        all_lines_in_file.pop_back();
        no_of_mappings++;
        
    }
    all_lines_in_file.pop_back();
}

void create_mapping(list<string>&mapping_data,unordered_map<string,unordered_set<int>>&map_table)
{
    for(auto i:mapping_data)
    {
        istringstream ss(i);
        string ts;
        ss>>ts;
        int ti;
        while(ss>>ti) map_table[ts].insert(ti);
        keys.insert(ts);
    }
    
}
int tc=0;
void decode(list<string>&all_lines_in_file,unordered_map<string,unordered_set<int>>&map_table,ofstream &writein)
{
    for(auto s: all_lines_in_file)
    {
        list<string>l;
        istringstream ss(s);
        string temp;
        int fl=1;
        while(ss>>temp)
        {
            if(keys.find(temp)!=keys.end())
            {
                for(auto i: map_table[temp])
                {
                    tc++;
                    writein<<i<<" ";
                }
            }
            else {writein<<temp<<" ";tc++;}
        }
        writein<<endl;
    }
}

int main(int argc, char* argv[])
{
    enc_file=argv[1];
    op_fi=argv[2];
    ofstream writein(op_fi);
    FILE* inFile = fopen(enc_file.c_str(), "r");
    list<string>all_lines_in_file;
    int all_items_in_data=0,no_lines=0;
    givealllines(inFile,all_lines_in_file,all_items_in_data,no_lines);
    list<string>mapping_data;
    int items_in_map=0,no_of_mappings=0;
    seperate_map_and_data(all_lines_in_file,mapping_data,items_in_map,no_of_mappings);
    unordered_map<string,unordered_set<int>>map_table;
    create_mapping(mapping_data,map_table);
    decode(all_lines_in_file,map_table,writein);
    writein.close();
    // cout<<"total items = "<<tc<<endl;
}