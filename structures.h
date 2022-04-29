#pragma once
// Simple linked list
template <class T>
class llist {
private:
	class lelem {
	public:
		lelem* next;
		T value;

		lelem() {
			next = nullptr;
		}

		lelem(T* avalue) {
			next = nullptr;
			value = *avalue;
		}
	};

	lelem* first;
	lelem* last;
public:
	llist() {
		first = nullptr;
		last = nullptr;
	}

	~llist() {
		lelem* pNext = first;
		lelem* pCur = first;
		while (pNext != nullptr) {
			pNext = pCur->next;
			delete(pCur);
			pCur = pNext;
		}
	}

	void add(T* value) {
		if (first == nullptr) {
			first = new lelem(value);
			last = first;
		}
		else {
			last->next = new lelem(value);
			last = last->next;
		}
	}
};