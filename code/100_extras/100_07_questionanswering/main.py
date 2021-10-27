from transformers import pipeline
from multiprocessing import freeze_support

if __name__ == '__main__':
	freeze_support()
	qa_pipeline = pipeline(
		"question-answering",
		model="deutsche-telekom/electra-base-de-squad2",
		tokenizer="deutsche-telekom/electra-base-de-squad2"
	)

	contexts = ['''Obamas Vater, Barack Hussein Obama Senior (1936–1982), stammte aus Nyang’oma Kogelo in Kenia und gehörte der Ethnie der Luo an. Obamas Mutter, Stanley Ann Dunham (1942–1995), stammte aus Wichita im US-Bundesstaat Kansas und hatte irische, britische, deutsche und Schweizer Vorfahren.[6] Obamas Eltern lernten sich als Studenten an der University of Hawaii at Manoa kennen. Sie heirateten 1961 in Hawaii, als Ann bereits schwanger war. Damals waren in anderen Teilen der USA Ehen zwischen Schwarzen und Weißen noch verboten. 1964 ließ sich das Paar scheiden. Der Vater setzte sein Studium an der Harvard University fort. Obama sah ihn als Zehnjähriger zum letzten Mal.

	Obama Senior starb im November 1982 in Nairobi an den Folgen eines Verkehrsunfalles. Barack Obama hat väterlicherseits drei ältere und drei jüngere Halbbrüder sowie die Halbschwester Auma, die in Deutschland studiert hat. Sein Halbbruder Malik lebt in Kenia und kandidierte 2013 erfolglos als Gouverneur des Siaya County.

	Die Mutter heiratete 1965 den Indonesier und späteren Ölmanager Lolo Soetoro. Nach dem Abschluss ihres Studiums zog sie mit ihrem Sohn Barack 1967 zu ihrem neuen Ehemann nach Jakarta in Indonesien, wo Obamas jüngere Halbschwester Maya geboren wurde. Dort besuchte Obama von 1967 bis 1970 die von Kapuzinern geführte St. Francis of Assisi Elementary School und 1970/71 eine staatliche, multireligiöse Schule. Während seiner Schulzeit in Indonesien verfasste Obama zwei Aufsätze mit dem Titel 'I want to become president'. Obamas Rufname in Bahasa Indonesia war Barry Soetoro, was beginnend mit dem Fall Berg v. Obama im Jahr 2008 immer wieder missbräuchlich und juristisch erfolglos als Indiz benutzt wurde, um Obamas Identität in Zweifel zu ziehen und seine Nominierung zum Präsidentschaftskandidaten zu verhindern. 1971 kehrte er nach Hawaii zurück, wo ihn seine Großeltern mütterlicherseits, Madelyn (1922–2008) und Stanley Dunham (1918–1992), aufzogen. Sie schulten ihn in die fünfte Klasse der privaten Punahou School ein. Seine Mutter kam mit seiner Halbschwester 1972 wieder nach Hawaii, um ihr Studium weiterzuführen, kehrte aber mit der Tochter 1975 nach Indonesien zurück, um ihre ethnologischen Forschungen fortzusetzen. Obama entschloss sich, nicht mitzugehen. Er blieb in Hawaii und schloss die Schule 1979 mit Auszeichnung ab. Er spielte in der Schule auch Basketball, zunächst in der Juniorenmannschaft und 1972 in der ersten Schulmannschaft. Damals konnte er sich vorstellen, Basketballprofi zu werden.

	Seine spätere Frau Michelle Robinson lernte Obama, zu der Zeit Student der Harvard Law School, 1989 während eines Sommerpraktikums in der Kanzlei Sidley Austin in Chicago kennen. Robinson, die als Anwältin in der Kanzlei arbeitete, war seine Betreuerin. Das Paar heiratete 1992 und hat zwei Töchter: Malia Ann (* 1998) und Natasha („Sasha“, * 2001). Michelle Obama war bis Ende 2008 in der öffentlichen Verwaltung von Chicago beschäftigt''']*2

	questions = ["Wie hieß seine Halbschwester?", 
				"Was wollte Obama mal werden?"]

	print(qa_pipeline(context=contexts, question=questions))