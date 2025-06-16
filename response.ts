export interface SelectResourceResponse {
  type: 'resource',
  resource: keyof Units,
}

export interface SelectResourcesResponse {
  type: 'resources',
  units: Units,
}

export interface SelectPolicyResponse {
  type: 'policy',
  policyId: PolicyId;
}

export interface SelectGlobalEventResponse {
  type: 'globalEvent',
  globalEventName: GlobalEventName;
}

export type AresGlobalParametersResponse = {
  lowOceanDelta: -1 | 0 | 1;
  highOceanDelta: -1 | 0 | 1;
  temperatureDelta: -1 | 0 | 1;
  oxygenDelta: -1 | 0 | 1;
}

export interface ShiftAresGlobalParametersResponse {
  type: 'aresGlobalParameters',
  response: AresGlobalParametersResponse;
}

export interface SelectProductionToLoseResponse {
  type: 'productionToLose',
  units: Units;
}

export interface SelectPaymentResponse {
  type: 'payment',
  payment: Payment;
}

export interface SelectColonyResponse {
  type: 'colony',
  colonyName: ColonyName;
}

export interface SelectAmountResponse {
  type: 'amount',
  amount: number;
}

export interface SelectDelegateResponse {
  type: 'delegate',
  player: ColorWithNeutral;
}

export interface SelectPartyResponse {
  type: 'party',
  partyName: PartyName;
}

export interface SelectPlayerResponse {
  type: 'player',
  player: ColorWithNeutral;
}
export interface SelectSpaceResponse {
  type: 'space',
  spaceId: SpaceId;
}
export const SPENDABLE_STANDARD_RESOURCES = [
  'megaCredits',
  'heat',
  'steel',
  'titanium',
  'plants',
] as const;


export const SPENDABLE_CARD_RESOURCES = [
  'microbes',
  'floaters',
  'lunaArchivesScience',
  'spireScience',
  'seeds',
  'auroraiData',
  'graphene',
  'kuiperAsteroids',
] as const;

export const OTHER_SPENDABLE_RESOURCES = [
  'corruption',
] as const;

export const SPENDABLE_RESOURCES = [...SPENDABLE_STANDARD_RESOURCES, ...SPENDABLE_CARD_RESOURCES, ...OTHER_SPENDABLE_RESOURCES] as const;
export type SpendableStandardResource = typeof SPENDABLE_STANDARD_RESOURCES[number];
export type SpendableCardResource = typeof SPENDABLE_CARD_RESOURCES[number];
export type OtherSpendableResource = typeof OTHER_SPENDABLE_RESOURCES[number];
export type SpendableResource = SpendableStandardResource | SpendableCardResource | OtherSpendableResource;
export type Payment = {[k in SpendableResource]: number};

export interface SelectProjectCardToPlayResponse {
  type: 'projectCard',
  card: CardName;
  payment: Payment;
}
export interface SelectCardResponse {
  type: 'card',
  cards: Array<CardName>;
}
export interface SelectInitialCardsResponse {
  type: 'initialCards',
  responses: Array<InputResponse>;
}
export interface AndOptionsResponse {
  type: 'and',
  responses: Array<InputResponse>;
}
export interface OrOptionsResponse {
  type: 'or',
  index: number;
  response: InputResponse;
}
export interface SelectOptionResponse {
  type: 'option',
}

export type InputResponse =
  AndOptionsResponse |
  OrOptionsResponse |
  SelectInitialCardsResponse |
  SelectAmountResponse |
  SelectCardResponse |
  SelectColonyResponse |
  SelectDelegateResponse |
  SelectOptionResponse |
  SelectPartyResponse |
  SelectPaymentResponse |
  SelectPlayerResponse |
  SelectProductionToLoseResponse |
  SelectProjectCardToPlayResponse |
  SelectSpaceResponse |
  ShiftAresGlobalParametersResponse |
  SelectGlobalEventResponse |
  SelectPolicyResponse |
  SelectResourceResponse |
  SelectResourcesResponse;